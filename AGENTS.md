# bhoga — Agent Contract

> **bhoga** (Sanskrit: *route, path, enjoyment of resources*) — a zero-server
> Python library that routes AI harnesses to the best available provider for a
> model, based on live subscription quota.

## For AI agents / harnesses

### Quick start

```python
from bhoga import Router, apply_to_hermes

router = Router()                          # background-inits quota discovery
rec = router.best_for("claude-opus-4")     # → RouterRecommendation | None
print(rec.hermes_model)                    # "anthropic/claude-opus-4"
print(rec.provider_id, rec.quota_pct)      # "claude_code" 0.80

# After each API call, record the turn
router.record_turn(
    pid="anthropic_api",
    model="claude-opus-4",
    input_tokens=1200,
    output_tokens=800,
    headers=dict(response.headers),
    status_code=response.status_code,
)

# Write recommendation into Hermes config
apply_to_hermes(router, "claude-opus-4")
```

### No LLM dependency

Quota discovery uses structured APIs and regex parsing — no LLM calls:

- **Claude Code**: reads `~/.claude/.credentials.json` → OAuth API (`/api/oauth/usage`)
- **Codex CLI**: runs `codex --non-interactive /status` → regex parse
- **API providers**: calibrated from HTTP response headers

### Dual cadence

Subscription providers (Claude Code, Codex) have two rate-limit windows:
- **burst** (5 hours) — short-term usage cap
- **weekly** (7 days) — longer-term usage cap

`best_for()` uses the **tighter** window (min remaining %) for routing decisions.

### Provider hierarchy

For each model family, bhoga tries providers in priority order:

| Family | Priority 1 | Priority 2 | Priority 3 |
|---|---|---|---|
| anthropic | `claude_code` | `anthropic_api` | `github-copilot` |
| openai | `openai-codex` | `openai_api` | `github-copilot` |

### Hermes provider mapping

| bhoga provider_id | Hermes canonical ID |
|---|---|
| `claude_code` | `anthropic` |
| `anthropic_api` | `anthropic` |
| `openai-codex` | `openai-codex` |
| `openai_api` | `openrouter` |
| `github-copilot` | `copilot` |

### State persistence

Quota state is persisted to `~/.cache/bhoga/state.json` (override with
`BHOGA_STATE` env var). Thread-safe via `threading.Lock`.

### Thread safety

All `Router` methods are thread-safe. Background threads handle:
- Initial quota discovery (`_init_bg`)
- Turn recording + header calibration (`record_turn`)
