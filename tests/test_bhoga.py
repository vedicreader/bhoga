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


# ── to_hermes_model ──────────────────────────────────────────────────────────

def test_hermes_model_string():
    assert to_hermes_model("claude_code", "claude-opus-4") == "anthropic/claude-opus-4"
    assert to_hermes_model("openai-codex", "gpt-4.1") == "openai-codex/gpt-4.1"
    assert to_hermes_model("github-copilot", "gpt-4.1") == "github-copilot/gpt-4.1"
    assert to_hermes_model("openai_api", "gpt-4.1") == "openrouter/gpt-4.1"


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


# ── Hermes integration ───────────────────────────────────────────────────────

def test_apply_to_hermes(tmp_path):
    """apply_to_hermes should write correct Hermes config."""
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
    assert config["model"] == "anthropic/claude-opus-4"
    assert config["compression"]["summary_provider"] == "anthropic"
    assert config["auxiliary"]["vision"]["provider"] == "anthropic"
    assert config["auxiliary"]["compression"]["provider"] == "anthropic"
