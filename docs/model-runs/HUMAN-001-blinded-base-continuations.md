# HUMAN-001 — Blinded Base-Model Continuations

- PR: [#50](https://github.com/Ayumu-J-S/llm_scratch/pull/50) (draft)
- Branch: `codex/human-001-blinded-eval`
- Ticket: `HUMAN-001`
- Hypothesis: two separated checkpoints from one RUN-001 launch can be compared
  by humans under a fixed, balanced, blinded continuation protocol without
  leaking checkpoint identity or reusing evaluation material for training.
- Experiment record: `docs/experiments/HUMAN-001-blinded-base-continuations.md`
- Started: 2026-07-19
- Final verdict: in progress; latest independent verdict is `FAIL`
- Record owner: HUMAN-001 implementation workflow

## Scope and decision context

- Goal: implement the HUMAN-001 preparation, blinding, scoring, unblinding,
  contamination, and evidence workflow. A real result remains blocked on
  RUN-001 checkpoints and two actual human reviewers.
- In scope: the committed bilingual prompt set, fixed generation contract,
  exact checkpoint/run lineage, complete prompt contamination scan, evaluator
  and runtime identity, private authenticated evidence, and agreement metrics.
- Out of scope: simulated ratings, model-generated ratings, training on any
  evaluation material, a model-quality conclusion before real ratings, or GPU
  performance claims from CPU validation.
- Relevant `PHILOSOPHY.md` principles: preserve research integrity, keep
  evaluation distinct from training, make base-model output explicit, and
  retain inspectable evidence for claims.
- Baseline commit/run: implementation began from BENCH-001 head
  `dcab48eee00eb82a195cc6d2cd9006bb62a8517f`; RUN-001 checkpoints are pending.
- Intended evidence: exact-head CPU quality gate, fixed-contract fixtures,
  complete contamination evidence, and independent `/review` cycles. No real
  HUMAN generation or rating result exists yet.

## Implementation and review timeline

Failed cycles are retained and must not be rewritten as passing cycles.

| Cycle | Phase | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | implementation | `c975147` through `c262b1a` | Implement prompt schema, blinded workflow, and path isolation | candidate | Added the balanced public/private workflow; supplemental audit found exact-artifact/path binding gaps | Supplemental `FAIL` on `c262b1a` |
| 2 | repair / review | `aee5a4e` | Bind evidence to exact artifacts | `FAIL` | Exact bundle/checkpoint/key binding improved; repository trust-root handling remained incomplete | Supplemental `FAIL` on `aee5a4e` |
| 3 | repair / review | `56c8f9fca46d238f64f36a3c5803b3d387f756b7` | Anchor evaluator trust to the repository | supplemental `PASS`; formal `FAIL` | Supplemental checks passed, but formal review found no complete prompt scan, no unique launch lineage, incomplete evaluator/runtime/config identity, and no fixed deterministic CUDA policy | Formal `/review` handoff on `56c8f9f` |
| 4 | repair | `bfc798a871f3d580c13e5e20a4e1791cbfc23a31`; target merges through `49e8580ba30f1f6de1174ddd01e43ccf750168ac` | Close every formal integrity gap | implemented | Added complete exact/NFC training-manifest scan, exact checkpoint-pair binding, unique inherited run lineage, shared BENCH/HUMAN determinism policy, complete prepare/import evaluator identity, 100 GB reserve, and regression coverage | Exact-head `make ci-cpu`: 516 passed, 1 skipped; config/lock/offline smoke passed |
| 5 | re-review | `49e8580ba30f1f6de1174ddd01e43ccf750168ac` against `origin/main` | Independent formal `/review` | `FAIL` | P2: context-limited generation could be published under the fixed 64-token contract. P2: reviewer ratings and their retained checksum came from separate file reads. | `codex review --base origin/main`; reviewer also reproduced 516 passed, 1 skipped |
| 6 | repair | successor to `49e8580` | Reject context truncation and bind parsed score bytes to their digest | in progress | Generation-result contract and one-read score evidence are being repaired with focused regression tests | Pending exact-head validation and independent re-review |

## Independent check selection and verdicts

### Review cycle 1 — formal review of `56c8f9f`

- Commit reviewed: `56c8f9fca46d238f64f36a3c5803b3d387f756b7`.
- Selected `CHECK.md` sections: applicable parts of 7 (direct change surface),
  8 (run identity, contamination, research integrity), and 9.1 (checkpoint and
  long-running evidence). DGX performance and real quality conclusions were
  N/A because no real evaluation run occurred.
- Ticket acceptance result: `FAIL`.
- Philosophy alignment: `FAIL` until contamination and evidence identity were
  fail-closed.
- Complexity / change-surface result: repair required shared runtime policy and
  one focused scanner, not a parallel configuration system.
- ML-system result: `FAIL`; evaluation provenance and leakage checks were not
  sufficient for a real HUMAN attempt.
- Verdict: `FAIL`.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| P1 | contamination | HUMAN prompts were not completely scanned as exact and normalized occurrences across checkpoint-owned training manifests | Formal review of `56c8f9f` | Scan before generation; block on any occurrence; bind and retain source/document/count/digest evidence without prompt text |
| P1 | launch lineage | Deterministic `experiment_id` identified a recipe, so independent same-recipe launches could masquerade as one run | Formal review of `56c8f9f` | Add unique run lineage to manifests/checkpoints, inherit it on resume, and require it across the checkpoint pair |
| P2 | evaluator identity | Private evidence did not bind dirty source bytes, complete resolved HUMAN config, lock, and concrete runtime/container identity | Formal review of `56c8f9f` | Capture and identity-bind complete evaluator evidence at preparation/import |
| P2 | determinism | HUMAN did not apply the fixed deterministic CUDA policy required for comparable evaluation | Formal review of `56c8f9f` | Share BENCH's strict math-SDPA/TF32/cuBLAS policy and retain it in evidence |

## Failed-review handoff — cycle 1

- From review cycle: formal review of `56c8f9f`.
- Failed check and why: CHECK 8 evaluation identity/contamination requirements
  were incomplete, and independent launches were not distinguishable.
- Reproduction command and evidence: inspect checkpoint pair identity and HUMAN
  preparation evidence at `56c8f9f`; no complete prompt scan was invoked before
  sampling.
- Relevant repository context and resolved Hydra config:
  `config/human_evaluation.yaml`, checkpoint-owned `resolved_config`, and the
  manifest-backed RUN-001 streaming profile.
- Invariants and constraints to preserve: fixed eight-prompt protocol, blinded
  public schema, same seed per prompt/checkpoint, private HMAC evidence, no
  simulated ratings, no GPU run before implementation passes review, and at
  least 100 GB free disk.
- Previous repair attempts: artifact/path/key and repository-root hardening on
  `c262b1a`, `aee5a4e`, and `56c8f9f`.
- Exact repair request: complete the four findings above and independently
  re-review the exact successor.
- Required completion evidence: focused invariants, full CPU quality gate,
  lock/diff checks, exact-head `/review`, and retained failed-cycle trail.

## Repair cycle 1

- Finding addressed: every cycle-1 finding.
- Change made: repair commit `bfc798a` added complete prompt scanning and
  evidence, unique inherited run lineage, shared deterministic evaluation, full
  evaluator identity, fixed pair validation, and 100 GB disk headroom.
- Validation rerun: exact `49e8580` CPU gate passed 516 tests with one skip,
  plus Hydra config, lock drift, and offline smoke checks.
- Remaining risk: independent re-review was still required.

### Review cycle 2 — formal re-review of `49e8580`

- Commit reviewed: `49e8580ba30f1f6de1174ddd01e43ccf750168ac`.
- Selected `CHECK.md` sections: the same applicable 7, 8, and 9.1 scope.
- Major sections marked N/A and why: no real DGX generation, human ratings,
  training change, performance claim, or quality conclusion.
- Ticket acceptance result: `FAIL`.
- Philosophy alignment: the design is aligned, but two evidence/contract races
  still block a real attempt.
- Complexity / change-surface result: no unnecessary abstraction finding.
- ML-system result: `FAIL`; published samples could be context-truncated, and a
  score checksum could identify bytes different from the parsed ratings.
- Verdict: `FAIL` because actionable findings remained.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| P2 | generation contract | `CheckpointSampler.generate` could return `context_limit`, while HUMAN retained only completion text and still advertised a fixed 64-token contract | `/review` lines 221-228 of `workflow.py` at `49e8580` | Reject any prompt/checkpoint whose allowed context is below 64 or whose result stops on context limit |
| P2 | score evidence | Ratings were parsed in one path read and hashed in a later path read, allowing replacement between them | `/review` lines 431-433 of `workflow.py` at `49e8580` | Parse and hash the exact same byte buffer |

## Failed-review handoff — cycle 2

- From review cycle: exact-head formal re-review of `49e8580`.
- Failed check and why: fixed evaluation-contract integrity and score-file
  traceability remained vulnerable to silent divergence.
- Reproduction command and evidence: `codex review --base origin/main`; inspect
  discarded `GenerationResult` metadata and the two independent score path reads.
- Relevant repository context and resolved Hydra config: fixed
  `generation.max_new_tokens=64`; authenticated private mapping; score imports
  from the private `reviews/` namespace.
- Invariants and constraints to preserve: no context-truncated public bundle,
  one exact checksum per parsed score buffer, private file/path checks, immutable
  result naming, no score simulation, and no GPU work.
- Previous repair attempts: cycle-1 repair `bfc798a`.
- Exact repair request: reject context-limit results and return score payload plus
  digest evidence from one read; add adversarial regression tests.
- Required completion evidence: focused HUMAN tests, full CPU gate, lock/diff
  checks, target-branch merge/revalidation, and independent exact-head re-review.

## Repair cycle 2

- Finding addressed: context truncation and score-read/checksum TOCTOU.
- Change made: in progress.
- Validation rerun: pending.
- Remaining risk: BENCH target has a separate narrow repair in flight; merge and
  exact-head revalidation are required before re-review.

## Independent re-review

- Commit reviewed: pending repair successor.
- Prior findings disposition: pending.
- New findings: pending.
- Verdict: pending.
- Evidence: pending.

## Merge authority and guarded audit

- Merge path: human review/merge by default; no merge is authorized from this
  incomplete review state.
- Human authorization and scope, or `N/A — human merge`: bounded roadmap series
  authorization exists, but review and exact-head gates still block merge.
- Exact reviewed head: latest reviewed head `49e8580`; failing.
- Final review verdict: `FAIL`.
- Actionable findings repaired and independently re-reviewed: no.
- Blocking review decision / newer human objection: actionable review findings
  remain until the successor passes independent re-review.
- Unresolved review threads: PR audit pending.
- Required-context and configured-workflow inventory: pending final audit.
- Exact-head check statuses: `49e8580` CPU gate passed; repair successor pending.
- Current base and mergeable evidence: target merge/revalidation pending.
- PR trail, validation, risks, and authorization parity: this record and PR #50
  must retain all failed cycles.
- Prohibited self-merge categories: no prohibited external action is requested,
  but failed review independently blocks merge.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: PR #50, pending.
- Immediate pre-merge refresh location and drift result: pending.
- Merge outcome: not merged.
