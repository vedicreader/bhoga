"""Tests for bhoga provider router."""
from __future__ import annotations

import json, tempfile, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest, yaml

from bhoga import (
    BillingCadence,
    ProviderQuota,
    ProviderStatus,
    QuotaWindow,
    Router,
    RouterRecommendation,
    apply_to_hermes,
    calibrate,
    get_codex_models,
    mark_throttled,
    model_family,
    to_hermes_model,
)
from bhoga.bhoga import blank, hierarchy_for, _ser, _deser, save_state, load_state


# ── model_family ─────────────────────────────────────────────────────────────

def test_model_family_anthropic():
    assert model_family("claude-opus-4-20250514") == "anthropic"
    assert model_family("sonnet-4") == "anthropic"

def test_model_family_openai():
    assert model_family("gpt-4.1") == "openai"
    assert model_family("o4-mini") == "openai"

def test_model_family_unknown():
    assert model_family("llama-3") == "other"


# ── hierarchy ────────────────────────────────────────────────────────────────

def test_hierarchy_anthropic():
    h = hierarchy_for("claude-opus-4-20250514")
    assert h == ["claude_code", "anthropic_api", "github-copilot"]

def test_hierarchy_openai():
    h = hierarchy_for("gpt-4.1")
    assert h == ["openai-codex", "openai_api", "github-copilot"]


# ── to_hermes_model / normalization ──────────────────────────────────────────

def test_hermes_model_string():
    # anthropic direct API → bare name with dots-to-hyphens, no vendor prefix
    assert to_hermes_model("claude_code",    "claude-opus-4")     == "claude-opus-4"
    assert to_hermes_model("anthropic_api",  "claude-sonnet-4.6") == "claude-sonnet-4-6"
    # openai-codex → bare name, dots preserved
    assert to_hermes_model("openai-codex",   "gpt-4.1")           == "gpt-4.1"
    # copilot → bare name, dots preserved
    assert to_hermes_model("github-copilot", "gpt-4.1")           == "gpt-4.1"
    assert to_hermes_model("github-copilot", "claude-sonnet-4.6") == "claude-sonnet-4.6"
    # openai_api → openrouter aggregator: vendor/model
    assert to_hermes_model("openai_api",     "gpt-4.1")           == "openai/gpt-4.1"

def test_hermes_model_strips_existing_vendor_prefix():
    """Vendor prefix already present should not be doubled."""
    assert to_hermes_model("openai_api", "openai/gpt-4.1") == "openai/gpt-4.1"

def test_hermes_model_anthropic_dots_to_hyphens():
    assert to_hermes_model("anthropic_api", "claude-sonnet-4.6") == "claude-sonnet-4-6"
    assert to_hermes_model("claude_code",   "claude-haiku-4.5")  == "claude-haiku-4-5"


# ── QuotaWindow ──────────────────────────────────────────────────────────────

def test_quota_window_remaining():
    w = QuotaWindow(name="burst", pct_used=30.0)
    assert w.pct_remaining == 70.0

def test_quota_window_clamped():
    w = QuotaWindow(name="burst", pct_used=110.0)
    assert w.pct_remaining == 0.0


# ── ProviderQuota.quota_pct ──────────────────────────────────────────────────

def test_quota_pct_windowed():
    """Tighter window governs."""
    now = datetime.now(timezone.utc)
    q = ProviderQuota(
        provider_id="claude_code", model_id="*",
        windows={
            "burst": QuotaWindow(name="burst", pct_used=80.0, updated_at=now),
            "weekly": QuotaWindow(name="weekly", pct_used=20.0, updated_at=now),
        },
    )
    # burst has 20% remaining, weekly has 80% remaining → min = 20%
    assert q.quota_pct == pytest.approx(0.20, abs=0.001)

def test_dual_cadence_tighter_window_governs():
    """When weekly is nearly exhausted, it should govern over burst."""
    now = datetime.now(timezone.utc)
    q = ProviderQuota(
        provider_id="claude_code", model_id="*",
        windows={
            "burst": QuotaWindow(name="burst", pct_used=10.0, updated_at=now),
            "weekly": QuotaWindow(name="weekly", pct_used=95.0, updated_at=now),
        },
    )
    # burst 90% remaining, weekly 5% remaining → min = 5%
    assert q.quota_pct == pytest.approx(0.05, abs=0.001)

def test_quota_pct_calibrated():
    q = ProviderQuota(provider_id="anthropic_api", model_id="claude-opus-4",
                      calibrated=800_000, ceiling=1_000_000)
    assert q.quota_pct == pytest.approx(0.80)

def test_quota_pct_consumed():
    q = ProviderQuota(provider_id="openai_api", model_id="gpt-4.1",
                      ceiling=500_000, consumed=100_000)
    assert q.quota_pct == pytest.approx(0.80)

def test_quota_pct_unknown():
    q = ProviderQuota(provider_id="test", model_id="m")
    assert q.quota_pct == -1.0


# ── is_usable / recompute ───────────────────────────────────────────────────

def test_exhausted_not_usable():
    q = ProviderQuota(provider_id="x", model_id="m",
                      status=ProviderStatus.EXHAUSTED)
    assert not q.is_usable()

def test_throttled_with_backoff():
    future = datetime.now(timezone.utc) + timedelta(minutes=5)
    q = ProviderQuota(provider_id="x", model_id="m",
                      status=ProviderStatus.THROTTLED, backoff_until=future)
    assert not q.is_usable()

def test_recompute_ok():
    q = ProviderQuota(provider_id="x", model_id="m", ceiling=1000, consumed=100)
    q.recompute()
    assert q.status == ProviderStatus.OK

def test_recompute_exhausted():
    q = ProviderQuota(provider_id="x", model_id="m", ceiling=1000, consumed=1000)
    q.recompute()
    assert q.status == ProviderStatus.EXHAUSTED

def test_recompute_low():
    q = ProviderQuota(provider_id="x", model_id="m", ceiling=1000, consumed=920)
    q.recompute()
    assert q.status == ProviderStatus.LOW


# ── Serialization ────────────────────────────────────────────────────────────

def test_round_trip_serialization():
    now = datetime.now(timezone.utc)
    q = ProviderQuota(
        provider_id="claude_code", model_id="*", priority=1,
        windows={"burst": QuotaWindow(name="burst", pct_used=50.0,
                                       window_min=300, updated_at=now)},
    )
    d = _ser(q)
    q2 = _deser(d)
    assert q2.provider_id == "claude_code"
    assert q2.windows["burst"].pct_used == 50.0

def test_state_save_load(tmp_path):
    p = tmp_path / "state.json"
    state = {"k": blank("anthropic_api", "m")}
    save_state(state, p)
    loaded = load_state(p)
    assert "k" in loaded
    assert loaded["k"].provider_id == "anthropic_api"


# ── calibrate ────────────────────────────────────────────────────────────────

def test_calibrate_openai():
    q = ProviderQuota(provider_id="openai_api", model_id="gpt-4.1", ceiling=1_000_000)
    calibrate(q, {"x-ratelimit-remaining-tokens": "600000"})
    assert q.calibrated == 600_000
    assert q.status == ProviderStatus.OK

def test_calibrate_anthropic():
    q = ProviderQuota(provider_id="anthropic_api", model_id="claude-opus-4",
                      ceiling=1_000_000)
    calibrate(q, {
        "anthropic-ratelimit-tokens-remaining": "50000",
        "anthropic-ratelimit-tokens-limit": "1000000",
    })
    assert q.calibrated == 50_000
    assert q.ceiling == 1_000_000
    assert q.status == ProviderStatus.LOW

def test_calibrate_copilot_request_headers():
    """Copilot calibration uses request-based headers."""
    q = ProviderQuota(provider_id="github-copilot", model_id="gpt-4.1",
                      ceiling=300)
    calibrate(q, {
        "x-ratelimit-remaining-requests": "250",
        "x-ratelimit-limit-requests": "300",
    })
    assert q.calibrated == 250
    assert q.ceiling == 300
    assert q.status == ProviderStatus.OK


# ── mark_throttled ───────────────────────────────────────────────────────────

def test_throttle_on_429():
    q = ProviderQuota(provider_id="x", model_id="m")
    mark_throttled(q, retry_after=120)
    assert q.status == ProviderStatus.THROTTLED
    assert q.backoff_until is not None
    assert not q.is_usable()


# ── Router ───────────────────────────────────────────────────────────────────

def test_highest_quota_wins(tmp_path):
    """Provider with most remaining quota should be selected."""
    p = tmp_path / "state.json"
    now = datetime.now(timezone.utc)
    state = {
        "claude_code:claude-opus-4": ProviderQuota(
            provider_id="claude_code", model_id="claude-opus-4", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=90.0,
                                           updated_at=now, window_min=300)},
        ),
        "anthropic_api:claude-opus-4": ProviderQuota(
            provider_id="anthropic_api", model_id="claude-opus-4", priority=2,
            ceiling=1_000_000, calibrated=800_000,
        ),
    }
    save_state(state, p)
    r = Router(state_path=p, eager=False)
    rec = r.best_for("claude-opus-4")
    assert rec is not None
    # anthropic_api has 80% remaining, claude_code burst has 10% → anthropic wins
    assert rec.provider_id == "anthropic_api"

def test_priority_tiebreak(tmp_path):
    """On equal quota, lower priority number wins."""
    p = tmp_path / "state.json"
    now = datetime.now(timezone.utc)
    state = {
        "claude_code:claude-opus-4": ProviderQuota(
            provider_id="claude_code", model_id="claude-opus-4", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=50.0,
                                           updated_at=now, window_min=300)},
        ),
        "anthropic_api:claude-opus-4": ProviderQuota(
            provider_id="anthropic_api", model_id="claude-opus-4", priority=2,
            ceiling=1_000_000, calibrated=500_000,
        ),
    }
    save_state(state, p)
    r = Router(state_path=p, eager=False)
    rec = r.best_for("claude-opus-4")
    assert rec is not None
    # Both ~50% remaining, priority 1 < priority 2 → claude_code wins
    assert rec.provider_id == "claude_code"

def test_exhausted_skipped(tmp_path):
    """Exhausted providers should be skipped."""
    p = tmp_path / "state.json"
    now = datetime.now(timezone.utc)
    state = {
        "claude_code:claude-opus-4": ProviderQuota(
            provider_id="claude_code", model_id="claude-opus-4", priority=1,
            ceiling=1000, consumed=1000,
            windows={"burst": QuotaWindow(name="burst", pct_used=100.0,
                                           updated_at=now, window_min=300)},
        ),
    }
    save_state(state, p)
    r = Router(state_path=p, eager=False)
    rec = r.best_for("claude-opus-4")
    assert rec is not None
    # Should skip exhausted claude_code and pick next available (optimistic blank)
    assert rec.provider_id != "claude_code"

def test_unknown_quota_routes_optimistically(tmp_path):
    """Unknown quota (-1) should still be routable."""
    p = tmp_path / "state.json"
    r = Router(state_path=p, eager=False)
    rec = r.best_for("claude-opus-4")
    assert rec is not None

def test_wildcard_fallback(tmp_path):
    """pid:* wildcard should be found when pid:specific_model is missing."""
    p = tmp_path / "state.json"
    now = datetime.now(timezone.utc)
    state = {
        "claude_code:*": ProviderQuota(
            provider_id="claude_code", model_id="*", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=20.0,
                                           updated_at=now, window_min=300)},
        ),
    }
    save_state(state, p)
    r = Router(state_path=p, eager=False)
    rec = r.best_for("claude-opus-4")
    assert rec is not None
    assert rec.provider_id == "claude_code"
    assert rec.quota_pct == pytest.approx(0.80, abs=0.01)

def test_record_turn_accumulates(tmp_path):
    """record_turn should update consumed tokens."""
    p = tmp_path / "state.json"
    r = Router(state_path=p, eager=False)
    # Prime state
    q = blank("openai_api", "gpt-4.1")
    q.ceiling = 1_000_000
    r._put(q)
    # Record directly via _record_bg (synchronous for test)
    r._record_bg("openai_api", "gpt-4.1", 500, 200, None, 200)
    q2 = r._get("openai_api", "gpt-4.1")
    assert q2 is not None
    assert q2.consumed == 700
    assert q2.n_requests == 1

def test_record_turn_429_throttles(tmp_path):
    """429 response should throttle the provider."""
    p = tmp_path / "state.json"
    r = Router(state_path=p, eager=False)
    q = blank("openai_api", "gpt-4.1")
    r._put(q)
    r._record_bg("openai_api", "gpt-4.1", 0, 0, {"retry-after": "30"}, 429)
    q2 = r._get("openai_api", "gpt-4.1")
    assert q2 is not None
    assert q2.status == ProviderStatus.THROTTLED
    assert not q2.is_usable()


# ── Router.set_quota ─────────────────────────────────────────────────────────

def test_set_quota_windowed_burst(tmp_path):
    """set_quota on a windowed provider sets burst window."""
    r = Router(state_path=tmp_path / "state.json", eager=False)
    r.set_quota("openai-codex", pct_remaining=75.0)
    q = r._get("openai-codex", "*")
    assert q is not None
    assert q.windows["burst"].pct_used == pytest.approx(25.0)
    assert q.quota_pct == pytest.approx(0.75, abs=0.01)
    assert q.status == ProviderStatus.OK

def test_set_quota_windowed_both_windows(tmp_path):
    """set_quota with weekly_pct sets both burst and weekly independently."""
    r = Router(state_path=tmp_path / "state.json", eager=False)
    r.set_quota("claude_code", pct_remaining=90.0, weekly_pct=60.0)
    q = r._get("claude_code", "*")
    assert q is not None
    assert q.windows["burst"].pct_used  == pytest.approx(10.0)
    assert q.windows["weekly"].pct_used == pytest.approx(40.0)
    # quota_pct = min(90%, 60%) / 100 = 0.60
    assert q.quota_pct == pytest.approx(0.60, abs=0.01)

def test_set_quota_windowed_weekly_defaults_to_burst(tmp_path):
    """If weekly_pct is omitted, it defaults to pct_remaining."""
    r = Router(state_path=tmp_path / "state.json", eager=False)
    r.set_quota("claude_code", pct_remaining=50.0)
    q = r._get("claude_code", "*")
    assert q.windows["burst"].pct_used  == pytest.approx(50.0)
    assert q.windows["weekly"].pct_used == pytest.approx(50.0)

def test_set_quota_token_based(tmp_path):
    """set_quota on a token/request-based provider sets calibrated."""
    r = Router(state_path=tmp_path / "state.json", eager=False)
    q0 = blank("github-copilot", "*")
    q0.ceiling = 500
    r._put(q0)
    r.set_quota("github-copilot", pct_remaining=80.0)
    q = r._get("github-copilot", "*")
    assert q is not None
    assert q.calibrated == 400   # 80% of 500
    assert q.status == ProviderStatus.OK

def test_set_quota_persists_to_disk(tmp_path):
    """set_quota should be persisted so a new Router instance can read it."""
    p = tmp_path / "state.json"
    r1 = Router(state_path=p, eager=False)
    r1.set_quota("openai-codex", pct_remaining=55.0)
    r2 = Router(state_path=p, eager=False)
    q = r2._get("openai-codex", "*")
    assert q is not None
    assert q.windows["burst"].pct_used == pytest.approx(45.0)

def test_set_quota_invalid_range(tmp_path):
    """set_quota should raise on out-of-range values."""
    r = Router(state_path=tmp_path / "state.json", eager=False)
    with pytest.raises(ValueError):
        r.set_quota("openai-codex", pct_remaining=110.0)
    with pytest.raises(ValueError):
        r.set_quota("openai-codex", pct_remaining=50.0, weekly_pct=-1.0)

def test_set_quota_affects_routing(tmp_path):
    """A provider with set_quota=0 should be treated as exhausted and skipped."""
    p = tmp_path / "state.json"
    r = Router(state_path=p, eager=False)
    r.set_quota("claude_code",  pct_remaining=0.0)
    r.set_quota("anthropic_api", pct_remaining=70.0)
    # Force anthropic_api into state with a ceiling so quota_pct is meaningful
    q = r._get("anthropic_api", "*")
    assert q is not None
    rec = r.best_for("claude-opus-4")
    assert rec is not None
    assert rec.provider_id == "anthropic_api"

def test_set_quota_specific_model(tmp_path):
    """set_quota can target a specific model key."""
    r = Router(state_path=tmp_path / "state.json", eager=False)
    r.set_quota("anthropic_api", pct_remaining=60.0, model="claude-opus-4")
    q = r._get("anthropic_api", "claude-opus-4")
    assert q is not None
    assert q.calibrated == 600_000  # 60% of default ceiling 1_000_000


# ── get_codex_models ─────────────────────────────────────────────────────────

def test_get_codex_models_returns_defaults():
    """Without local files present, returns the hardcoded default list."""
    from bhoga.bhoga import _CODEX_DEFAULT_MODELS
    models = get_codex_models()
    assert isinstance(models, list)
    assert len(models) > 0
    for mid in _CODEX_DEFAULT_MODELS:
        assert mid in models

def test_get_codex_models_reads_cache(tmp_path):
    """models_cache.json is read when available."""
    cache = tmp_path / "models_cache.json"
    cache.write_text(json.dumps({"models": [
        {"slug": "gpt-5.4", "priority": 1, "supported_in_api": True},
        {"slug": "gpt-5.4-mini", "priority": 2, "supported_in_api": True},
        {"slug": "hidden-model", "priority": 3, "visibility": "hide"},
    ]}))
    from bhoga import bhoga as bhoga_mod
    orig = bhoga_mod._DEFAULTS["providers"]["openai-codex"]["models_cache"]
    bhoga_mod._DEFAULTS["providers"]["openai-codex"]["models_cache"] = str(cache)
    try:
        models = get_codex_models()
        assert "gpt-5.4" in models
        assert "gpt-5.4-mini" in models
        assert "hidden-model" not in models
    finally:
        bhoga_mod._DEFAULTS["providers"]["openai-codex"]["models_cache"] = orig

def test_get_codex_models_no_duplicates():
    """No model ID should appear twice."""
    models = get_codex_models()
    assert len(models) == len(set(models))


# ── Hermes integration ───────────────────────────────────────────────────────

def test_apply_to_hermes(tmp_path):
    """apply_to_hermes writes dict model format with provider + api_mode."""
    state_path = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    now = datetime.now(timezone.utc)

    state = {
        "claude_code:*": ProviderQuota(
            provider_id="claude_code", model_id="*", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=20.0,
                                           updated_at=now, window_min=300)},
        ),
    }
    save_state(state, state_path)
    r = Router(state_path=state_path, eager=False)

    ok = apply_to_hermes(r, "claude-opus-4", config_path=config_path)
    assert ok

    config = yaml.safe_load(config_path.read_text())
    # model is now a dict with provider and api_mode
    assert isinstance(config["model"], dict)
    assert config["model"]["default"]  == "claude-opus-4"
    assert config["model"]["provider"] == "anthropic"
    assert config["model"]["api_mode"] == "anthropic_messages"
    assert config["compression"]["summary_provider"] == "anthropic"
    # auxiliary NOT written by default
    assert "auxiliary" not in config

def test_apply_to_hermes_write_auxiliary(tmp_path):
    """write_auxiliary=True populates all auxiliary task providers."""
    state_path = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    now = datetime.now(timezone.utc)

    state = {
        "claude_code:*": ProviderQuota(
            provider_id="claude_code", model_id="*", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=20.0,
                                           updated_at=now, window_min=300)},
        ),
    }
    save_state(state, state_path)
    r = Router(state_path=state_path, eager=False)

    ok = apply_to_hermes(r, "claude-opus-4", config_path=config_path,
                         write_auxiliary=True)
    assert ok

    config = yaml.safe_load(config_path.read_text())
    assert config["auxiliary"]["vision"]["provider"] == "anthropic"
    assert config["auxiliary"]["compression"]["provider"] == "anthropic"

def test_apply_to_hermes_codex_api_mode(tmp_path):
    """openai-codex provider should produce api_mode=codex_responses."""
    state_path = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    now = datetime.now(timezone.utc)

    state = {
        "openai-codex:*": ProviderQuota(
            provider_id="openai-codex", model_id="*", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=10.0,
                                           updated_at=now, window_min=300)},
        ),
    }
    save_state(state, state_path)
    r = Router(state_path=state_path, eager=False)

    ok = apply_to_hermes(r, "gpt-5.4", config_path=config_path)
    assert ok

    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["provider"] == "openai-codex"
    assert config["model"]["api_mode"] == "codex_responses"
    assert config["model"]["default"]  == "gpt-5.4"

def test_apply_to_hermes_copilot_stub(tmp_path):
    """Copilot provider should write a providers.copilot stub."""
    state_path = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    now = datetime.now(timezone.utc)

    state = {
        "github-copilot:*": ProviderQuota(
            provider_id="github-copilot", model_id="*", priority=3,
            ceiling=300, calibrated=280,
        ),
    }
    save_state(state, state_path)
    r = Router(state_path=state_path, eager=False)

    ok = apply_to_hermes(r, "gpt-4.1", config_path=config_path)
    assert ok

    config = yaml.safe_load(config_path.read_text())
    # Hermes canonical ID for Copilot is "copilot"
    assert config["model"]["provider"] == "copilot"
    assert config["model"]["api_mode"] == "chat_completions"
    # Non-destructive stub under providers.copilot
    assert "copilot" in config.get("providers", {})
    stub = config["providers"]["copilot"]
    assert stub["base_url"] == "https://api.githubcopilot.com"
    assert stub["api_key_env"] == "COPILOT_GITHUB_TOKEN"

def test_apply_to_hermes_copilot_stub_not_overwritten(tmp_path):
    """Existing providers.copilot entry should not be overwritten."""
    state_path  = tmp_path / "state.json"
    config_path = tmp_path / "config.yaml"
    now = datetime.now(timezone.utc)

    # Pre-existing config with a custom copilot stub
    config_path.write_text(yaml.dump({
        "providers": {"copilot": {"base_url": "https://my-proxy.example.com"}},
    }))

    state = {
        "github-copilot:*": ProviderQuota(
            provider_id="github-copilot", model_id="*", priority=3,
            ceiling=300, calibrated=200,
        ),
    }
    save_state(state, state_path)
    r = Router(state_path=state_path, eager=False)
    apply_to_hermes(r, "gpt-4.1", config_path=config_path)

    config = yaml.safe_load(config_path.read_text())
    # Should preserve the user's custom URL
    assert config["providers"]["copilot"]["base_url"] == "https://my-proxy.example.com"


# ── reset_if_due ─────────────────────────────────────────────────────────────

def test_reset_if_due_fires_on_past_resets_at():
    """A window whose resets_at is in the past should be zeroed out and True returned."""
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    q = ProviderQuota(
        provider_id="claude_code", model_id="*",
        windows={
            "burst": QuotaWindow(name="burst", pct_used=90.0, resets_at=past,
                                 window_min=300, updated_at=past),
        },
    )
    fired = q.reset_if_due()
    assert fired is True
    assert q.windows["burst"].pct_used == 0.0
    assert q.windows["burst"].resets_at is None
    assert q.status == ProviderStatus.OK  # window reset → 100% remaining → OK

def test_reset_if_due_does_not_fire_for_future():
    """A window whose resets_at is in the future must not be reset."""
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    q = ProviderQuota(
        provider_id="claude_code", model_id="*",
        windows={
            "burst": QuotaWindow(name="burst", pct_used=80.0, resets_at=future,
                                 window_min=300, updated_at=datetime.now(timezone.utc)),
        },
    )
    fired = q.reset_if_due()
    assert fired is False
    assert q.windows["burst"].pct_used == 80.0

def test_reset_if_due_persists_via_router(tmp_path):
    """best_for() should call _save() when a window is reset, so the next Router sees it."""
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    p = tmp_path / "state.json"
    state = {
        "claude_code:claude-opus-4": ProviderQuota(
            provider_id="claude_code", model_id="claude-opus-4", priority=1,
            windows={"burst": QuotaWindow(name="burst", pct_used=90.0, resets_at=past,
                                          window_min=300, updated_at=past)},
        ),
    }
    save_state(state, p)
    r = Router(state_path=p, eager=False)
    r.best_for("claude-opus-4")  # triggers reset_if_due + _save

    # A fresh router loading the same file should see pct_used=0
    r2 = Router(state_path=p, eager=False)
    q = r2._get("claude_code", "claude-opus-4")
    assert q is not None
    assert q.windows["burst"].pct_used == 0.0


# ── _parse_claude_cli text parsing ───────────────────────────────────────────

def test_parse_claude_cli_text_burst():
    """_parse_claude_cli should extract the burst window from text output."""
    from bhoga.bhoga import _parse_claude_cli
    from unittest.mock import patch, MagicMock
    mock_result = MagicMock()
    mock_result.stdout = "Usage: 72% used in last 5 hours (burst window)"
    with patch("subprocess.run", return_value=mock_result):
        windows = _parse_claude_cli()
    assert windows is not None
    assert "burst" in windows
    assert windows["burst"].pct_used == pytest.approx(72.0)

def test_parse_claude_cli_text_weekly():
    """_parse_claude_cli should detect the weekly window when context contains 'week'."""
    from bhoga.bhoga import _parse_claude_cli
    from unittest.mock import patch, MagicMock
    mock_result = MagicMock()
    mock_result.stdout = "Weekly quota: 45% used this week"
    with patch("subprocess.run", return_value=mock_result):
        windows = _parse_claude_cli()
    assert windows is not None
    assert "weekly" in windows
    assert windows["weekly"].pct_used == pytest.approx(45.0)

def test_parse_claude_cli_text_remaining():
    """'remaining' direction should invert: 30% remaining → 70% used."""
    from bhoga.bhoga import _parse_claude_cli
    from unittest.mock import patch, MagicMock
    mock_result = MagicMock()
    mock_result.stdout = "30% remaining in burst window"
    with patch("subprocess.run", return_value=mock_result):
        windows = _parse_claude_cli()
    assert windows is not None
    assert windows["burst"].pct_used == pytest.approx(70.0)

def test_parse_claude_cli_no_output():
    """Empty stdout should return None."""
    from bhoga.bhoga import _parse_claude_cli
    from unittest.mock import patch, MagicMock
    mock_result = MagicMock()
    mock_result.stdout = ""
    with patch("subprocess.run", return_value=mock_result):
        assert _parse_claude_cli() is None


# ── parse_codex_quota JSON path ───────────────────────────────────────────────

def test_parse_codex_quota_json_burst():
    """parse_codex_quota should parse JSON with a burst key."""
    from bhoga.bhoga import parse_codex_quota
    from unittest.mock import patch, MagicMock
    payload = json.dumps({"burst": {"pct_used": 55.0}})
    mock_result = MagicMock()
    mock_result.stdout = payload
    with patch("subprocess.run", return_value=mock_result):
        windows = parse_codex_quota()
    assert windows is not None
    assert "burst" in windows
    assert windows["burst"].pct_used == pytest.approx(55.0)

def test_parse_codex_quota_json_utilization():
    """parse_codex_quota should convert utilization (0-1) to pct_used (0-100)."""
    from bhoga.bhoga import parse_codex_quota
    from unittest.mock import patch, MagicMock
    payload = json.dumps({"five_hour": {"utilization": 0.8}, "seven_day": {"utilization": 0.3}})
    mock_result = MagicMock()
    mock_result.stdout = payload
    with patch("subprocess.run", return_value=mock_result):
        windows = parse_codex_quota()
    assert windows is not None
    assert windows["burst"].pct_used  == pytest.approx(80.0)
    assert windows["weekly"].pct_used == pytest.approx(30.0)

def test_parse_codex_quota_text_fallback():
    """parse_codex_quota falls back to text parse when output is not JSON."""
    from bhoga.bhoga import parse_codex_quota
    from unittest.mock import patch, MagicMock
    mock_result = MagicMock()
    mock_result.stdout = "gpt-5.4 high · 60% left · ~4h"
    with patch("subprocess.run", return_value=mock_result):
        windows = parse_codex_quota()
    assert windows is not None
    assert windows["burst"].pct_used == pytest.approx(40.0)  # 100 - 60


# ── Router(eager=True) smoke test ─────────────────────────────────────────────

def test_router_eager_init_smoke(tmp_path):
    """Router with eager=True should complete background init without errors."""
    from unittest.mock import patch
    p = tmp_path / "state.json"
    # Patch out all external I/O so the test is hermetic
    with patch("bhoga.bhoga.check_quota", return_value=None), \
         patch("bhoga.bhoga.fetch_models", return_value=[]):
        r = Router(state_path=p, eager=True)
        # Give background thread time to complete
        r._worker.join(timeout=0)          # worker is always running, don't block
        import threading, time
        for _ in range(50):                # wait up to 5 s for _init_bg to finish
            if not any(t.name.startswith("Thread") and t.is_alive()
                       for t in threading.enumerate()
                       if t is not threading.current_thread() and t is not r._worker):
                break
            time.sleep(0.1)
        rec = r.best_for("claude-opus-4")
        assert rec is not None            # optimistic blank quota always routes
        r.close()

