"""bhoga — zero-server provider routing based on live subscription quota.

Routes AI harnesses to the best available provider for a model by tracking
quota across subscription plans (Claude Code, Codex CLI) and pay-as-you-go
APIs (Anthropic, OpenAI, GitHub Models).  First-class Hermes engine support.

Design: fastai/fastcore style — succinct, annotated, no ceremony.
"""
from __future__ import annotations

import json, logging, os, re, subprocess, threading, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import httpx
import yaml
from dateutil.parser import parse as dt_parse
from fastcore.basics import ifnone, store_attr

log = logging.getLogger(__name__)

# ── Enums ────────────────────────────────────────────────────────────────────

class BillingCadence(StrEnum):
    HOURLY  = auto()
    DAILY   = auto()
    WEEKLY  = auto()
    MONTHLY = auto()

class ProviderStatus(StrEnum):
    OK        = auto()
    LOW       = auto()
    THROTTLED = auto()
    EXHAUSTED = auto()
    UNKNOWN   = auto()

# ── Defaults ─────────────────────────────────────────────────────────────────

_FAM: dict[str, str] = {
    "claude": "anthropic", "opus": "anthropic", "sonnet": "anthropic", "haiku": "anthropic",
    "gpt": "openai", "o1": "openai", "o3": "openai", "o4": "openai", "codex": "openai",
    "gemini": "google", "palm": "google",
}

_DEFAULTS: dict[str, Any] = {
    "hierarchy": {
        "anthropic": ["claude_code", "anthropic_api", "github-copilot"],
        "openai":    ["openai-codex", "openai_api", "github-copilot"],
        "google":    [],
        "other":     ["github-copilot"],
    },
    "providers": {
        "claude_code": {
            "family": "anthropic",
            "priority": 1,
            "quota_source": "oauth_api",
            "oauth_url": "https://api.anthropic.com/api/oauth/usage",
            "cred_path": "~/.claude/.credentials.json",
            "cred_key": "claudeAiOauth.accessToken",
            "windows": {
                "burst":  {"field": "five_hour",  "default_minutes": 300},
                "weekly": {"field": "seven_day",  "default_minutes": 10080},
            },
        },
        "anthropic_api": {
            "family": "anthropic",
            "priority": 2,
            "cadence": "monthly",
            "models_url": "https://api.anthropic.com/v1/models",
            "auth_env": "ANTHROPIC_API_KEY",
            "hdr_rem": "anthropic-ratelimit-tokens-remaining",
            "hdr_limit": "anthropic-ratelimit-tokens-limit",
            "hdr_rst": "anthropic-ratelimit-tokens-reset",
        },
        "openai-codex": {
            "family": "openai",
            "priority": 1,
            "quota_source": "cli_parse",
            "cli_cmd": ["codex", "--non-interactive", "/status"],
            "windows": {
                "burst":  {"default_minutes": 300},
                "weekly": {"default_minutes": 10080},
            },
        },
        "openai_api": {
            "family": "openai",
            "priority": 2,
            "cadence": "monthly",
            "models_url": "https://api.openai.com/v1/models",
            "auth_env": "OPENAI_API_KEY",
            "hdr_rem": "x-ratelimit-remaining-tokens",
            "hdr_rst": "x-ratelimit-reset-tokens",
        },
        "github-copilot": {
            "family": "universal",
            "priority": 3,
            "cadence": "hourly",
            "models_url": "https://models.github.ai/catalog/models",
            "auth_env": "GITHUB_TOKEN",
            "hdr_rem": "x-ratelimit-remaining-tokens",
            "hdr_rst": "x-ratelimit-reset",
        },
    },
    "hermes_provider": {
        "claude_code":    "anthropic",
        "anthropic_api":  "anthropic",
        "openai-codex":   "openai-codex",
        "openai_api":     "openrouter",
        "github-copilot": "github-copilot",
    },
}

STATE_PATH = Path(os.environ.get("BHOGA_STATE", Path.home() / ".cache" / "bhoga" / "state.json"))

# ── Provider helpers ─────────────────────────────────────────────────────────

def model_family(m: str) -> str:
    """Map a model name to its family via prefix match."""
    return next((f for p, f in _FAM.items() if m.lower().startswith(p)), "other")

def hierarchy_for(model: str) -> list[str]:
    """Ordered provider IDs to try for *model*."""
    return list(_DEFAULTS["hierarchy"].get(model_family(model), _DEFAULTS["hierarchy"]["other"]))

def cfg(pid: str) -> dict[str, Any]:
    """Provider config dict from defaults."""
    return _DEFAULTS["providers"].get(pid, {})

def to_hermes_model(pid: str, mid: str) -> str:
    """Convert bhoga provider+model to Hermes `provider/model` string."""
    hp = _DEFAULTS["hermes_provider"].get(pid, "openrouter")
    return f"{hp}/{mid}"

# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class QuotaWindow:
    """One rate-limit window (e.g., 5h burst or 7d weekly)."""
    name:       str
    pct_used:   float              = 0.0
    resets_at:  datetime | None    = None
    window_min: int                = 300
    updated_at: datetime | None    = None

    @property
    def pct_remaining(self) -> float:
        return max(0.0, 100.0 - self.pct_used)


@dataclass
class ProviderQuota:
    """Tracks quota for one provider×model pair."""
    provider_id:   str
    model_id:      str
    plan:          str              = "unknown"
    ceiling:       int | None       = None
    cadence:       BillingCadence   = BillingCadence.MONTHLY
    reset_at:      datetime | None  = None
    consumed:      int              = 0
    n_requests:    int              = 0
    calibrated:    int | None       = None
    calibrated_at: datetime | None  = None
    status:        ProviderStatus   = ProviderStatus.UNKNOWN
    backoff_until: datetime | None  = None
    priority:      int              = 99
    windows:       dict[str, QuotaWindow] = field(default_factory=dict)

    @property
    def quota_pct(self) -> float:
        """Remaining quota as fraction [0,1]. -1 = unknown."""
        if self.windows:
            updated = [w.pct_remaining for w in self.windows.values() if w.updated_at]
            return min(updated) / 100.0 if updated else -1.0
        if self.calibrated is not None and self.ceiling:
            return max(0.0, self.calibrated / self.ceiling)
        if self.ceiling:
            return max(0.0, 1.0 - self.consumed / self.ceiling)
        return -1.0

    def is_usable(self) -> bool:
        now = datetime.now(timezone.utc)
        if self.backoff_until and now < self.backoff_until:
            return False
        return self.status not in (ProviderStatus.EXHAUSTED, ProviderStatus.THROTTLED)

    def recompute(self) -> None:
        """Derive status from current quota numbers."""
        pct = self.quota_pct
        if pct < 0:
            self.status = ProviderStatus.UNKNOWN
        elif pct <= 0.0:
            self.status = ProviderStatus.EXHAUSTED
        elif pct < 0.10:
            self.status = ProviderStatus.LOW
        else:
            self.status = ProviderStatus.OK

    def reset_if_due(self) -> None:
        """Reset counters if cadence window has elapsed."""
        now = datetime.now(timezone.utc)
        # Window-based providers: reset individual windows
        for w in self.windows.values():
            if w.resets_at and now >= w.resets_at:
                w.pct_used = 0.0
                w.resets_at = None
                w.updated_at = now
        # Token-based providers: reset consumed
        if self.reset_at and now >= self.reset_at:
            self.consumed = 0
            self.calibrated = self.ceiling
            self.reset_at = None
        self.recompute()


@dataclass
class RouterRecommendation:
    """What `best_for()` returns."""
    provider_id:  str
    model_id:     str
    quota_pct:    float
    status:       ProviderStatus
    hermes_model: str
    priority:     int


def blank(pid: str, mid: str = "*") -> ProviderQuota:
    """Create a fresh ProviderQuota with defaults from config."""
    c = cfg(pid)
    cadence = BillingCadence(c.get("cadence", "monthly"))
    windows: dict[str, QuotaWindow] = {}
    for wname, wcfg in c.get("windows", {}).items():
        windows[wname] = QuotaWindow(name=wname, window_min=wcfg.get("default_minutes", 300))
    return ProviderQuota(
        provider_id=pid, model_id=mid, cadence=cadence,
        priority=c.get("priority", 99), windows=windows,
    )

# ── Serialization ────────────────────────────────────────────────────────────

def _ser_dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None

def _deser_dt(s: str | None) -> datetime | None:
    return dt_parse(s) if s else None

def _ser_window(w: QuotaWindow) -> dict:
    return {"name": w.name, "pct_used": w.pct_used, "window_min": w.window_min,
            "resets_at": _ser_dt(w.resets_at), "updated_at": _ser_dt(w.updated_at)}

def _deser_window(d: dict) -> QuotaWindow:
    return QuotaWindow(name=d["name"], pct_used=d.get("pct_used", 0.0),
                       window_min=d.get("window_min", 300),
                       resets_at=_deser_dt(d.get("resets_at")),
                       updated_at=_deser_dt(d.get("updated_at")))

def _ser(q: ProviderQuota) -> dict:
    return {
        "provider_id": q.provider_id, "model_id": q.model_id, "plan": q.plan,
        "ceiling": q.ceiling, "cadence": q.cadence.value, "consumed": q.consumed,
        "n_requests": q.n_requests, "calibrated": q.calibrated, "priority": q.priority,
        "status": q.status.value,
        "reset_at": _ser_dt(q.reset_at), "calibrated_at": _ser_dt(q.calibrated_at),
        "backoff_until": _ser_dt(q.backoff_until),
        "windows": {k: _ser_window(v) for k, v in q.windows.items()},
    }

def _deser(d: dict) -> ProviderQuota:
    return ProviderQuota(
        provider_id=d["provider_id"], model_id=d.get("model_id", "*"),
        plan=d.get("plan", "unknown"), ceiling=d.get("ceiling"),
        cadence=BillingCadence(d.get("cadence", "monthly")),
        consumed=d.get("consumed", 0), n_requests=d.get("n_requests", 0),
        calibrated=d.get("calibrated"), priority=d.get("priority", 99),
        status=ProviderStatus(d.get("status", "unknown")),
        reset_at=_deser_dt(d.get("reset_at")),
        calibrated_at=_deser_dt(d.get("calibrated_at")),
        backoff_until=_deser_dt(d.get("backoff_until")),
        windows={k: _deser_window(v) for k, v in d.get("windows", {}).items()},
    )

def save_state(state: dict[str, ProviderQuota], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: _ser(v) for k, v in state.items()}, indent=2))

def load_state(path: Path = STATE_PATH) -> dict[str, ProviderQuota]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {k: _deser(v) for k, v in raw.items()}
    except Exception:
        log.warning("bhoga: corrupt state at %s, starting fresh", path)
        return {}

# ── Header calibration ───────────────────────────────────────────────────────

def calibrate(q: ProviderQuota, headers: dict[str, str]) -> ProviderQuota:
    """Update quota from HTTP response headers."""
    c = cfg(q.provider_id)
    hdr_rem = c.get("hdr_rem", "")
    hdr_limit = c.get("hdr_limit", "")
    hdr_rst = c.get("hdr_rst", "")
    now = datetime.now(timezone.utc)

    if hdr_rem and hdr_rem in headers:
        try:
            q.calibrated = int(headers[hdr_rem])
            q.calibrated_at = now
        except (ValueError, TypeError):
            pass
    if hdr_limit and hdr_limit in headers:
        try:
            q.ceiling = int(headers[hdr_limit])
        except (ValueError, TypeError):
            pass
    if hdr_rst and hdr_rst in headers:
        raw = headers[hdr_rst]
        try:
            # Unix timestamp (GitHub) or duration string
            ts = float(raw)
            q.reset_at = datetime.fromtimestamp(ts, tz=timezone.utc)
        except ValueError:
            try:
                q.reset_at = dt_parse(raw)
            except Exception:
                pass
    q.recompute()
    return q


def mark_throttled(q: ProviderQuota, retry_after: float = 60.0) -> ProviderQuota:
    """Mark provider as throttled (e.g., on 429)."""
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    q.status = ProviderStatus.THROTTLED
    q.backoff_until = now + timedelta(seconds=retry_after)
    return q

# ── Quota discovery ──────────────────────────────────────────────────────────

def fetch_models(pid: str) -> list[str]:
    """List model IDs available from a provider's API."""
    c = cfg(pid)
    url = c.get("models_url")
    auth_env = c.get("auth_env", "")
    if not url:
        return []
    key = os.environ.get(auth_env, "")
    if not key:
        return []
    headers: dict[str, str] = {}
    if pid == "github-copilot":
        headers = {"Authorization": f"Bearer {key}",
                    "X-GitHub-Api-Version": "2026-03-10"}
    elif pid == "anthropic_api":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"Authorization": f"Bearer {key}"}
    try:
        r = httpx.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        # OpenAI / Anthropic: {"data": [{"id": ...}]}
        if isinstance(data, dict) and "data" in data:
            return [m["id"] for m in data["data"] if "id" in m]
        # GitHub Models: [{"id": ...}]
        if isinstance(data, list):
            return [m["id"] for m in data if "id" in m]
    except Exception as e:
        log.debug("bhoga: fetch_models(%s) failed: %s", pid, e)
    return []


def parse_claude_quota() -> dict[str, QuotaWindow] | None:
    """Read Claude Code quota from OAuth API or CLI fallback. Returns windows dict or None."""
    # Try OAuth API first
    cred_path = Path(cfg("claude_code").get("cred_path", "~/.claude/.credentials.json")).expanduser()
    token = _read_claude_token(cred_path)
    if token:
        windows = _fetch_claude_oauth(token)
        if windows:
            return windows
    # Fallback: parse CLI output
    return _parse_claude_cli()


def _read_claude_token(cred_path: Path) -> str | None:
    if not cred_path.exists():
        return None
    try:
        data = json.loads(cred_path.read_text())
        # Navigate nested key: "claudeAiOauth.accessToken"
        parts = cfg("claude_code").get("cred_key", "claudeAiOauth.accessToken").split(".")
        obj: Any = data
        for p in parts:
            obj = obj[p]
        return str(obj) if obj else None
    except Exception:
        return None


def _fetch_claude_oauth(token: str) -> dict[str, QuotaWindow] | None:
    """Fetch structured quota from Anthropic OAuth usage API."""
    url = cfg("claude_code").get("oauth_url", "https://api.anthropic.com/api/oauth/usage")
    now = datetime.now(timezone.utc)
    try:
        r = httpx.get(url, headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("bhoga: claude oauth failed: %s", e)
        return None

    windows: dict[str, QuotaWindow] = {}
    wcfg = cfg("claude_code").get("windows", {})
    for wname, wdef in wcfg.items():
        api_field = wdef.get("field", "")
        default_min = wdef.get("default_minutes", 300)
        if api_field and api_field in data:
            w_data = data[api_field]
            pct_used = float(w_data.get("utilization", 0)) * 100.0
            resets_raw = w_data.get("resets_at")
            resets_at = dt_parse(resets_raw) if resets_raw else None
            windows[wname] = QuotaWindow(
                name=wname, pct_used=pct_used, resets_at=resets_at,
                window_min=default_min, updated_at=now,
            )
    return windows if windows else None


_CLAUDE_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(used|left|remaining)", re.IGNORECASE)
_CLAUDE_WINDOW_RE = re.compile(r"(session|5.?h(?:our)?|burst|week|7.?d(?:ay)?)", re.IGNORECASE)

def _parse_claude_cli() -> dict[str, QuotaWindow] | None:
    """Fallback: parse `claude /usage` stdout."""
    try:
        result = subprocess.run(
            ["claude", "-p", "/usage"], capture_output=True, text=True, timeout=30,
        )
        text = result.stdout
    except Exception:
        return None
    if not text:
        return None

    now = datetime.now(timezone.utc)
    windows: dict[str, QuotaWindow] = {}
    # Look for percentage patterns near window labels
    for m in _CLAUDE_PCT_RE.finditer(text):
        pct_val = float(m.group(1))
        direction = m.group(2).lower()
        pct_used = pct_val if "used" in direction else (100.0 - pct_val)
        # Determine which window by scanning context
        context = text[max(0, m.start() - 80):m.end() + 20].lower()
        if any(w in context for w in ("week", "7d", "7-d", "seven")):
            windows["weekly"] = QuotaWindow(name="weekly", pct_used=pct_used,
                                            window_min=10080, updated_at=now)
        else:
            windows["burst"] = QuotaWindow(name="burst", pct_used=pct_used,
                                           window_min=300, updated_at=now)
    return windows if windows else None


_CODEX_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(used|left|remaining)", re.IGNORECASE)

def parse_codex_quota() -> dict[str, QuotaWindow] | None:
    """Parse Codex CLI /status output for dual-cadence windows."""
    c = cfg("openai-codex")
    cmd = c.get("cli_cmd", ["codex", "--non-interactive", "/status"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        text = result.stdout
    except Exception:
        return None
    if not text:
        return None

    now = datetime.now(timezone.utc)
    windows: dict[str, QuotaWindow] = {}
    for m in _CODEX_PCT_RE.finditer(text):
        pct_val = float(m.group(1))
        direction = m.group(2).lower()
        pct_used = pct_val if "used" in direction else (100.0 - pct_val)
        context = text[max(0, m.start() - 80):m.end() + 20].lower()
        if any(w in context for w in ("week", "7d", "7-d")):
            windows["weekly"] = QuotaWindow(name="weekly", pct_used=pct_used,
                                            window_min=10080, updated_at=now)
        else:
            windows["burst"] = QuotaWindow(name="burst", pct_used=pct_used,
                                           window_min=300, updated_at=now)
    return windows if windows else None


def check_quota(pid: str) -> dict[str, QuotaWindow] | None:
    """Dispatch quota check for subscription-based providers."""
    if pid == "claude_code":
        return parse_claude_quota()
    if pid == "openai-codex":
        return parse_codex_quota()
    return None

# ── Router ───────────────────────────────────────────────────────────────────

class Router:
    """Stateful provider router.  Thread-safe, background-init, no LLM dependency."""

    def __init__(self, *, state_path: Path | None = None, eager: bool = True):
        self._lock = threading.Lock()
        self._path = ifnone(state_path, STATE_PATH)
        with self._lock:
            self._state: dict[str, ProviderQuota] = load_state(self._path)
        if eager:
            threading.Thread(target=self._init_bg, daemon=True).start()

    # ── key helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _key(pid: str, mid: str) -> str:
        return f"{pid}:{mid}"

    def _get(self, pid: str, mid: str) -> ProviderQuota | None:
        """Look up quota: exact match, then wildcard fallback."""
        with self._lock:
            q = self._state.get(self._key(pid, mid))
            if q is None:
                q = self._state.get(self._key(pid, "*"))
            return q

    def _put(self, q: ProviderQuota) -> None:
        with self._lock:
            self._state[self._key(q.provider_id, q.model_id)] = q

    def _save(self) -> None:
        with self._lock:
            save_state(dict(self._state), self._path)

    # ── background init ──────────────────────────────────────────────────

    def _init_bg(self) -> None:
        """Discover models and subscription quotas in background."""
        # Subscription providers: check quota windows
        for pid in ("claude_code", "openai-codex"):
            try:
                windows = check_quota(pid)
                if windows:
                    q = self._get(pid, "*") or blank(pid)
                    q.windows = windows
                    q.recompute()
                    self._put(q)
            except Exception as e:
                log.debug("bhoga: bg init %s failed: %s", pid, e)

        # API providers: list models
        for pid in ("anthropic_api", "openai_api", "github-copilot"):
            try:
                models = fetch_models(pid)
                for mid in models:
                    key = self._key(pid, mid)
                    with self._lock:
                        if key not in self._state:
                            self._state[key] = blank(pid, mid)
            except Exception as e:
                log.debug("bhoga: bg init %s failed: %s", pid, e)
        self._save()

    # ── routing ──────────────────────────────────────────────────────────

    def best_for(self, model: str) -> RouterRecommendation | None:
        """Pick the best provider for *model* based on live quota."""
        candidates: list[ProviderQuota] = []
        for pid in hierarchy_for(model):
            q = self._get(pid, model)
            if q is None:
                # Create optimistic blank
                q = blank(pid, model)
                self._put(q)
            q.reset_if_due()
            if q.is_usable():
                candidates.append(q)

        if not candidates:
            return None

        # Sort: highest remaining quota first, priority breaks ties (lower = better)
        candidates.sort(key=lambda q: (q.quota_pct, -q.priority), reverse=True)
        best = candidates[0]
        return RouterRecommendation(
            provider_id=best.provider_id,
            model_id=model if best.model_id == "*" else best.model_id,
            quota_pct=best.quota_pct,
            status=best.status,
            hermes_model=to_hermes_model(best.provider_id, model),
            priority=best.priority,
        )

    # ── recording ────────────────────────────────────────────────────────

    def record_turn(self, pid: str, model: str, input_tokens: int = 0,
                    output_tokens: int = 0, headers: dict[str, str] | None = None,
                    status_code: int = 200) -> None:
        """Record a completed API turn. Runs calibration in background."""
        threading.Thread(
            target=self._record_bg, daemon=True,
            args=(pid, model, input_tokens, output_tokens, headers, status_code),
        ).start()

    def _record_bg(self, pid: str, model: str, inp: int, out: int,
                   headers: dict[str, str] | None, status_code: int) -> None:
        q = self._get(pid, model)
        if q is None:
            q = blank(pid, model)
        q.consumed += inp + out
        q.n_requests += 1
        if headers:
            calibrate(q, headers)
        if status_code == 429:
            retry_after = float((headers or {}).get("retry-after", "60"))
            mark_throttled(q, retry_after)
        else:
            q.recompute()
        self._put(q)
        self._save()

    # ── introspection ────────────────────────────────────────────────────

    def quotas(self) -> dict[str, ProviderQuota]:
        with self._lock:
            return dict(self._state)

# ── Hermes integration ───────────────────────────────────────────────────────

def apply_to_hermes(router: Router, model: str,
                    config_path: str | Path | None = None) -> bool:
    """Write the best provider for *model* into Hermes config.yaml.

    Updates ``model``, ``compression.summary_provider``, and auxiliary
    provider fields to use the recommended bhoga provider.

    Returns True if config was written, False on failure.
    """
    rec = router.best_for(model)
    if rec is None:
        log.warning("bhoga: no provider available for %s", model)
        return False

    path = Path(ifnone(config_path, Path.home() / ".hermes" / "config.yaml"))
    if path.exists():
        config = yaml.safe_load(path.read_text()) or {}
    else:
        config = {}

    hermes_pid = _DEFAULTS["hermes_provider"].get(rec.provider_id, "openrouter")

    # Set main model
    config["model"] = rec.hermes_model

    # Set compression summary provider
    compression = config.setdefault("compression", {})
    compression["summary_provider"] = hermes_pid

    # Set auxiliary providers
    auxiliary = config.setdefault("auxiliary", {})
    for task in ("vision", "web_extract", "compression", "session_search",
                 "skills_hub", "approval", "mcp", "flush_memories"):
        task_cfg = auxiliary.setdefault(task, {})
        task_cfg["provider"] = hermes_pid

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    log.info("bhoga: wrote Hermes config → %s (provider=%s)", path, hermes_pid)
    return True
