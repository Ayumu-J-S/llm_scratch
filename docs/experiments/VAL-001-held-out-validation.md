# VAL-001 — Shared Held-Out Validation Scoring

- Roadmap ticket: `VAL-001`
- Branch: `codex/val-001-held-out-validation`
- Draft PR: [#42](https://github.com/Ayumu-J-S/llm_scratch/pull/42)
- Experiment owner: implementation agent; exact runtime model/reasoning are not exposed
- Status: implementation complete; independent heavy review and DGX evidence blocked
- Started (UTC): 2026-07-13
- Last updated (UTC): 2026-07-13T12:05:00Z
- Model-run provenance: `docs/model-runs/VAL-001-held-out-validation.md`

## Predeclared question and decision rule

- Hypothesis (one falsifiable claim): one token-weighted causal-LM scorer can
  produce identical held-out NLL and perplexity from the training-time and
  standalone checkpoint paths while retaining immutable checkpoint, manifest,
  corpus, and evaluated-token identities.
- Expected result: the two paths agree exactly on a deterministic CPU fixture;
  known logits match analytically calculated NLL; Japanese, English, and
  aggregate denominators reconcile; overlap and memorization misuse fail before
  scoring.
- Success condition: every VAL-001 acceptance criterion and its ticket-required
  known-logit, parity, cadence, overlap, and checkpoint-milestone check passes.
- Failure condition / stop condition: stop on any score-path divergence, identity
  omission, denominator mismatch, train/validation overlap, or result that labels
  same-corpus memorization as held-out validation.
- Relevant baseline commit and run: DATA-004 stacked head
  `e1d4ed8af98de84a3393cd0f6e517f9daf649138`; DATA-004 evidence is recorded in
  `docs/experiments/DATA-004-pinned-baseline-mixture.md`.
- Baseline metrics and evidence link: existing training-time validation behavior
  and tests on the stacked base; no standalone VAL-001 score exists yet.
- Smallest run capable of answering the question: deterministic CPU fixture
  checkpoint evaluation plus focused and full network-free tests.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware | N/A — correctness ticket uses CPU fixture | Ticket acceptance criteria |
| Training tokens | Fixture only; no consequential pretraining run | Scoring parity, not model quality |
| Optimizer steps | Small deterministic fixture checkpoint only | Checkpoint milestone acceptance |
| Evaluation work and cadence | One fixed bounded validation pass per trigger | VAL-001 scope |
| Checkpoint count and bytes | Minimum fixture checkpoint(s) | Standalone/training parity |
| Local / external / W&B storage | Local JSON; W&B disabled in evidence run | Optional summary remains off by default |
| DGX Spark BF16 smoke | 50–200 steps on the real DATA-004 path; record validation pause, evaluated targets/s, step median/p95, data wait, memory, and pre/post validation throughput recovery; prefer validation-off/on A/B | CHECK.md §6.3; target hardware/data availability is an explicit gate, never N/A |

## Attempt 1 — implementation and CPU fixture evidence

### Launch identity

- Started / ended (UTC): 2026-07-13T12:01:10Z / 2026-07-13T12:01:11Z
- Outcome: CPU correctness and training-time/standalone parity passed; real DGX
  BF16 evidence is blocked by the installed CPU-only Torch runtime.
- Exact command: see [`VAL-001-cpu-parity.json`](evidence/VAL-001-cpu-parity.json)
- Fully resolved Hydra configuration: `/tmp/val001-cpu-a852/resolved_config.yaml`
  (SHA-256 `6f1616500b513a0555ad6dd097af000e420dbc910694dcae3f33f0c275d99c3d`)
- Standalone output: `/tmp/val001-eval-a852/result.json` (SHA-256
  `b379b152e2df3b98981af5c2551577348f81b5e6e05d96c5ef2a95bcfef470e6`)
- Git commit SHA: `a8520d7fad718574d1fca4293e6f969c7a478b79`
- Worktree state: clean at the implementation commit when the evidence ran
- Dependency lock identity: `uv.lock`, SHA-256
  `a02dec14fc9a20e5314eae368e9e22a289616791f55cd81b68ceadd25b71ad91`
- Container/image identity: `N/A — CPU fixture used; no container launched`

### Scientific identity

- Model architecture/config identity and parameter count: random-initialized
  `SimpleDecoderTransformer`, embed 8, heads 2, layers 1, dropout 0, 860,562
  parameters
- Initialization / pretrained-weight check: random-initialized project model only
- Tokenizer name, immutable revision/checksum, and special-token contract:
  canonical LLM-jp tokenizer, fingerprint
  `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`
- Training/validation manifest identity/checksum: fixture manifest
  `00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31`, dataset
  fingerprint `21c82c527fb8fafbbba4e2ea2bdf7057aed48ec8ac995a369a356747c70cd05b`;
  this is intentionally same-corpus memorization smoke, not held-out evidence
- Train/validation disjointness evidence: production streaming preflight still
  calls `validate_disjoint_manifests`; the CPU attempt used the explicitly
  permitted same-corpus memorization profile and was namespace-separated
- Random seeds: Hydra seed 42; deterministic CPU algorithms enabled; validation
  loader factory derives the fixed validation stream from the validation seed
- Hardware identity: `gx10-02db`, aarch64, NVIDIA GB10, driver `580.159.03`;
  PyTorch `2.10.0+cpu`, CUDA unavailable
- Software/runtime identity: Python 3.11.15, lock SHA above, commit above
- Precision and numerical controls: FP32 deterministic fixture

### Counters, evidence, and integrity

- Actual elapsed time, optimizer steps, training tokens, examples, and target
  tokens: one optimizer step; 64 training target tokens; 72 evaluated targets;
  9 fixed windows
- Training/validation/evaluation metrics with denominators: aggregate NLL
  `11.207522365781996`, perplexity `73682.63004369408`; training-time and
  standalone scores matched exactly
- Evaluation identities: window SHA-256
  `6cbbf181afc2cd44d4ce9859c49afb962d58c1a8ff8f729c1fa85dd2ba0d3f8c`; target
  SHA-256 `5390413b4a8d6b982b31cae770df3856b7bda402dfe38d25bcd89690c9b97be3`
- System metrics: CPU fixture only; training-time validation pause
  `0.01193903700914234s`, standalone pause `0.008372592012165114s`; no DGX
  throughput/memory claim is made
- W&B run/project/artifact IDs: `N/A — W&B disabled`
- Checkpoint identity: best logical checkpoint at step 1 / 64 training tokens;
  physical SHA-256
  `10695b16bd84e8fe68bf8aa4e56fb90d57203c1d3963a59ed1c49afb9be4bd05`,
  10,365,425 bytes
- Logs and immutable evidence: [`VAL-001-cpu-parity.json`](evidence/VAL-001-cpu-parity.json)
- Integrity checks: no raw held-out text/tokens recorded; batching-independent
  digests matched; memorization metrics used only `memorization/*`
- Comparison with the predeclared baseline: baseline had aggregate-only
  training-time validation and no standalone result; this attempt demonstrates
  the new parity/identity contract on the fixed CPU fixture

### Attempt interpretation

- Result against success/failure conditions: CPU scorer, attribution, cadence,
  mode restoration, milestone identity, and standalone parity conditions pass;
  the CHECK §6.3 real-path condition remains blocked
- Failure or anomaly: the target host exposes a GB10 but the locked environment
  contains CPU-only Torch (`torch.cuda.is_available() == False`, CUDA build
  `None`, BF16 `False`); the real DATA-004 path was not launched
- Most likely cause and supporting evidence: environment diagnostic and
  `nvidia-smi` output show the hardware/driver, while PyTorch reports no CUDA
  runtime; this is a runtime packaging blocker, not a scorer result
- Alternatives ruled out and how: network-free full suite and CPU parity passed;
  no real-path performance conclusion is inferred from those checks
- What remains uncertain: 50–200-step DGX BF16 validation pause, targets/s,
  step median/p95, data wait, memory, and post-validation throughput recovery
  must be measured after a CUDA-capable pinned runtime and DATA-004 sources are
  available

## DGX Spark CHECK §6.3 gate

- Status: `blocked / PASS WITH NOTE candidate`, not N/A
- Required command: a bounded 50–200-step BF16 smoke on the real DATA-004
  profile with fixed validation windows, plus validation-off/on comparison when
  practical
- Attempted precondition evidence: `uv run --no-sync python
  scripts/diagnose_environment.py` reported `gx10-02db`, `aarch64`, GB10,
  driver `580.159.03`, but `PyTorch 2.10.0+cpu`, `CUDA: available=False`, and
  `BF16=False`; `nvidia-smi` independently reported `NVIDIA GB10`
- Missing measurements: validation pause, evaluated targets/s, step median/p95,
  data wait, memory, and throughput recovery on the real path
- Next operator action: install/use the repository's pinned CUDA-capable DGX
  runtime, rerun the bounded smoke with the exact resolved Hydra config and
  identities, then append the measurements without rewriting this blocked
  attempt

## Conclusion

- Hypothesis result: supported on the deterministic CPU fixture; overall ticket
  handoff remains blocked pending the independent heavier review and DGX §6.3
  measurement
- Evidence-backed conclusion: one scorer now produces token-weighted aggregate
  and per-corpus results, stable fixed-window/token identities, and exact
  training-time/standalone parity. Same-corpus smoke is visibly
  `memorization/*` and cannot masquerade as held-out validation.
- Uncertainty and limitations: the real DATA-004 BF16 path was unavailable in
  this locked environment; VAL-001 remains stacked on DATA-004 and will require
  retarget/rebase after that dependency merges
- Exactly one next step or next question: provide a CUDA-capable pinned DGX
  runtime and run the bounded §6.3 smoke for the missing performance evidence.
