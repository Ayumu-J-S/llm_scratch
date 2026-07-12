# Codex model provenance

This repository records model provenance without guessing values hidden by the
Codex runtime. Use `scripts/capture_model_provenance.py` at the start of each
implementation, review, repair, and re-review phase, and attach the JSON output
to the model-run record.

## Two separate namespaces

`requested` is configuration or invocation context. It may contain a value from
`~/.codex/config.toml` (currently `gpt-5.6-sol` and `xhigh`) or an explicit
command-line request. It is not evidence of the active deployment.

`actual` is limited to values visibly supplied by the active runtime display.
The current runtime displays the product as `Codex` and the family as `GPT-5`,
but the exact deployment/model identifier and reasoning mode are not exposed in
this session. Those fields must be recorded as `not exposed by runtime` with an
unavailable reason. Never translate “Sol” into an exact model ID or “Ultra” into
`xhigh` without an explicit runtime source.

Source precedence is: active runtime display, explicit invocation metadata,
configuration defaults, then unavailable. A delegated agent owns its own
provenance; the parent agent's model must not be copied into the child record.

## Safe capture

```bash
python scripts/capture_model_provenance.py \
  --phase implementation --role agent --task-path /root/prov-001 \
  --requested-model gpt-5.6-sol --requested-reasoning-mode xhigh \
  --actual-product Codex --actual-model-family GPT-5 \
  --output /tmp/prov-001.json
```

The command records UTC time, branch, commit, and the installed `codex --version`
when available. By default it records no raw thread ID. `--include-thread-id`
stores only a short SHA-256 digest of `CODEX_THREAD_ID`; it never writes the raw
ID. Prompts, hidden chain-of-thought, token counts, and secrets are never read
or emitted. Missing Git or CLI metadata is represented as unavailable rather
than causing a false identity claim.

For every model-run record, copy the JSON's requested and actual sections into
the execution trail and keep failed or blocked captures. Historical records are
not rewritten when this contract changes.
