"""bhoga — zero-server provider routing based on live subscription quota.

Routes AI harnesses to the best available provider for a model by tracking
quota across subscription plans (Claude Code, Codex CLI) and pay-as-you-go
APIs (Anthropic, OpenAI, GitHub Models).  First-class Hermes engine support.

Design: fastai/fastcore style — succinct, annotated, no ceremony.
"""
from __future__ import annotations

import copy, json, logging, os, queue, re, subprocess, threading, time
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
            "cadence": "monthly",   # premium requests reset 1st of month UTC
            "base_url":  "https://api.githubcopilot.com",
            "models_url": "https://api.githubcopilot.com/models",
            "auth_env": "GITHUB_TOKEN",
            "hdr_rem":   "x-ratelimit-remaining-requests",
            "hdr_limit": "x-ratelimit-limit-requests",
            "hdr_rst":   "x-ratelimit-reset-requests",
        },
    },
    "hermes_provider": {
        "claude_code":    "anthropic",
        "anthropic_api":  "anthropic",
        "openai-codex":   "openai-codex",
        "openai_api":     "openai",
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
                    "X-GitHub-Api-Version": "2022-11-28"}
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
    """Parse Codex CLI /status output for dual-cadence windows.

    Tries JSON first (forward-compatible), then falls back to best-effort
    text parse of the TUI status line (e.g. "gpt-5.4 high · 100% left · ~").
    Weekly window is always UNKNOWN from the CLI — calibrate it via
    ``Router.set_quota()`` or header calibration if you need precision.
    """
    c = cfg("openai-codex")
    cmd = c.get("cli_cmd", ["codex", "--non-interactive", "/status"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        text = result.stdout
    except Exception:
        return None
    if not text:
        return None

    # Try structured JSON first (forward-compatible for when Codex adds it)
    try:
        data = json.loads(text)
        windows: dict[str, QuotaWindow] = {}
        now = datetime.now(timezone.utc)
        if "burst" in data or "five_hour" in data:
            burst = data.get("burst") or data.get("five_hour", {})
            pct_used = float(burst.get("pct_used", burst.get("utilization", 0))) * (
                1 if burst.get("pct_used") is not None else 100)
            windows["burst"] = QuotaWindow(name="burst", pct_used=pct_used,
                                           window_min=300, updated_at=now)
        if "weekly" in data or "seven_day" in data:
            weekly = data.get("weekly") or data.get("seven_day", {})
            pct_used = float(weekly.get("pct_used", weekly.get("utilization", 0))) * (
                1 if weekly.get("pct_used") is not None else 100)
            windows["weekly"] = QuotaWindow(name="weekly", pct_used=pct_used,
                                            window_min=10080, updated_at=now)
        if windows:
            return windows
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Best-effort text parse — weekly window not available from CLI text output
    now = datetime.now(timezone.utc)
    windows = {}
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
        # Single worker thread drains all record_turn calls serially — prevents
        # unbounded thread spawning and eliminates per-turn TOCTOU races.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
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
        """Atomically persist state: snapshot under lock, then rename-replace."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = {k: _ser(v) for k, v in self._state.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2))
        tmp.rename(self._path)  # atomic on POSIX

    # ── background init ──────────────────────────────────────────────────

    def _init_bg(self) -> None:
        """Discover models and subscription quotas in background."""
        # Subscription providers: check quota windows
        for pid in ("claude_code", "openai-codex"):
            try:
                windows = check_quota(pid)
                if windows:
                    with self._lock:
                        key = self._key(pid, "*")
                        q = self._state.get(key) or blank(pid, "*")
                        q.windows = windows
                        q.recompute()
                        self._state[key] = q
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
            # Atomic get-or-create + reset under a single lock acquisition
            with self._lock:
                key = self._key(pid, model)
                q = self._state.get(key) or self._state.get(self._key(pid, "*"))
                if q is None:
                    q = blank(pid, model)
                    self._state[key] = q
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
        """Record a completed API turn.

        Non-blocking: enqueues the work item and returns immediately.
        The single background worker serialises all writes, eliminating
        TOCTOU races and unbounded thread creation.
        """
        self._queue.put((pid, model, input_tokens, output_tokens, headers, status_code))

    def _worker_loop(self) -> None:
        """Drain the record queue; runs on a single persistent daemon thread."""
        while True:
            item = self._queue.get()
            if item is None:  # None is the shutdown sentinel
                break
            pid, model, inp, out, headers, status_code = item
            self._record_bg(pid, model, inp, out, headers, status_code)

    def _record_bg(self, pid: str, model: str, inp: int, out: int,
                   headers: dict[str, str] | None, status_code: int) -> None:
        """Called only from the worker thread — no concurrent invocations."""
        with self._lock:
            key = self._key(pid, model)
            q = (self._state.get(key)
                 or self._state.get(self._key(pid, "*"))
                 or blank(pid, model))
            # Shallow-copy scalar fields before releasing lock so calibrate/
            # mark_throttled can run without holding it.
            q = copy.copy(q)
            q.windows = dict(q.windows)   # copy windows mapping too
            q.consumed += inp + out
            q.n_requests += 1

        if headers:
            calibrate(q, headers)
        if status_code == 429:
            retry_after = float((headers or {}).get("retry-after", "60"))
            mark_throttled(q, retry_after)
        else:
            q.recompute()

        with self._lock:
            self._state[self._key(pid, model)] = q
        self._save()

    # ── manual quota override ─────────────────────────────────────────────

    def set_quota(self, pid: str, pct_remaining: float,
                  weekly_pct: float | None = None,
                  model: str = "*") -> None:
        """Manually set the remaining quota for a provider.

        Use this as a fallback when automatic discovery is unavailable —
        for example when Codex CLI is not installed, when running in CI, or
        when the user already knows their current quota state.

        Args:
            pid:           Provider ID — "claude_code", "openai-codex", or
                           "github-copilot" (also works for "anthropic_api" /
                           "openai_api").
            pct_remaining: Percentage of quota remaining [0 .. 100].
                           For windowed providers this sets the *burst* window.
            weekly_pct:    Optional separate weekly-window remaining percentage.
                           Only meaningful for "claude_code" and "openai-codex".
                           Defaults to *pct_remaining* when omitted.
            model:         Model key to associate with (default ``"*"`` = all
                           models for this provider).

        Examples::

            router.set_quota("openai-codex", pct_remaining=75.0)
            router.set_quota("claude_code",  pct_remaining=90.0, weekly_pct=60.0)
            router.set_quota("github-copilot", pct_remaining=85.0)
        """
        if not (0.0 <= pct_remaining <= 100.0):
            raise ValueError(f"pct_remaining must be 0–100, got {pct_remaining}")
        if weekly_pct is not None and not (0.0 <= weekly_pct <= 100.0):
            raise ValueError(f"weekly_pct must be 0–100, got {weekly_pct}")

        now = datetime.now(timezone.utc)
        c = cfg(pid)

        with self._lock:
            key = self._key(pid, model)
            q = self._state.get(key) or blank(pid, model)
            q = copy.copy(q)
            q.windows = dict(q.windows)

            if c.get("windows"):
                # Window-based provider (claude_code, openai-codex)
                wcfg = c["windows"]
                burst_min  = wcfg.get("burst",  {}).get("default_minutes", 300)
                weekly_min = wcfg.get("weekly", {}).get("default_minutes", 10080)

                burst_win = q.windows.get("burst") or QuotaWindow(
                    name="burst", window_min=burst_min)
                burst_win = copy.copy(burst_win)
                burst_win.pct_used  = 100.0 - pct_remaining
                burst_win.updated_at = now
                q.windows["burst"] = burst_win

                if "weekly" in wcfg:
                    weekly_win = q.windows.get("weekly") or QuotaWindow(
                        name="weekly", window_min=weekly_min)
                    weekly_win = copy.copy(weekly_win)
                    weekly_win.pct_used  = 100.0 - (
                        weekly_pct if weekly_pct is not None else pct_remaining)
                    weekly_win.updated_at = now
                    q.windows["weekly"] = weekly_win
            else:
                # Token/request-based provider (anthropic_api, openai_api, github-copilot)
                ceiling = q.ceiling or 1_000_000
                q.calibrated    = int(ceiling * pct_remaining / 100.0)
                q.calibrated_at = now

            q.recompute()
            self._state[key] = q

        self._save()

    # ── introspection ────────────────────────────────────────────────────

    def quotas(self) -> dict[str, ProviderQuota]:
        with self._lock:
            return dict(self._state)

# ── Hermes integration ───────────────────────────────────────────────────────

def apply_to_hermes(router: Router, model: str,
                    config_path: str | Path | None = None,
                    write_auxiliary: bool = False) -> bool:
    """Write the best provider for *model* into Hermes config.yaml.

    Always updates:
    - ``model`` — the ``provider/model`` string Hermes should use
    - ``compression.summary_provider`` — keeps summarisation in sync

    Optionally (``write_auxiliary=True``) also overwrites all auxiliary
    task providers (vision, web_extract, compression, …).

    For GitHub Copilot, writes a non-destructive provider stub so Hermes
    knows the base URL and auth env var.

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

    # Always set main model
    config["model"] = rec.hermes_model

    # Always set compression summary provider
    compression = config.setdefault("compression", {})
    compression["summary_provider"] = hermes_pid

    # Optional: overwrite all auxiliary task providers
    if write_auxiliary:
        auxiliary = config.setdefault("auxiliary", {})
        for task in ("vision", "web_extract", "compression", "session_search",
                     "skills_hub", "approval", "mcp", "flush_memories"):
            task_cfg = auxiliary.setdefault(task, {})
            task_cfg["provider"] = hermes_pid

    # Non-destructive Copilot provider stub so Hermes knows the endpoint
    if rec.provider_id == "github-copilot":
        providers = config.setdefault("providers", {})
        if "github-copilot" not in providers:
            providers["github-copilot"] = {
                "type": "openai_compatible",
                "base_url": cfg("github-copilot").get("base_url",
                                "https://api.githubcopilot.com"),
                "api_key_env": cfg("github-copilot").get("auth_env", "GITHUB_TOKEN"),
            }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    log.info("bhoga: wrote Hermes config → %s (provider=%s)", path, hermes_pid)
    return True
