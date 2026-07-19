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
| 6 | repair | `6128ad2d185fe87f295576c68020c1d700ccdc78`; target merge through `00c404521089d7694608ee55ec2b1ddaa985dea1` | Reject context truncation and bind parsed score bytes to their digest | implemented | Rejected context-limited samples and parsed/hashed each score from one byte buffer | Focused HUMAN gate passed 23 tests; exact-head CPU gate passed 519 tests with 1 skip |
| 7 | re-review | `00c404521089d7694608ee55ec2b1ddaa985dea1` against `origin/main` | Independent formal `/review` | `FAIL` | P1: cached prompt scans omitted transitive producer identity. P2: operational/volatile evidence changed assignment identity. P2: prompt parsing and hashing used separate reads. | `codex review --base origin/main`; reviewer independently reran 519 passed, 1 skipped |
| 8 | repair | `f92dfe97cf46a3ce805523a6243259ef9aeed6af` | Close all cycle-7 identity and byte-binding gaps | implemented | Complete producer identity, stable assignment inputs, and one-buffer prompt parsing implemented with adversarial regressions | Focused HUMAN gate passed 25 tests; exact-head CPU gate passed 521 tests with 1 skip |
| 9 | re-review | `f6ca50bf1e43ce4e68f7c237be78e43554c629f8` against `origin/main` | Independent formal `/review` | `FAIL` | P1: a fresh launch could inherit an occupied run directory's lineage. P2: HUMAN scanned the corpus before rejecting known device/precision/context incompatibility. | `codex review --base origin/main`; reviewer independently inspected the broad branch and confirmed the suite passed |
| 10 | repair | `774d1bf9e29c8f6adf7c049447cfc63d3d1b5bdc` through `b6f03f9606867b36a7165878ab192c2c29f4c6b4` | Close both cycle-9 fail-closed ordering gaps | implemented | Fresh launches reject existing manifests, preparation is serialized across same-directory collisions, explicit resumes must supply verified lineage, and HUMAN device/precision/tokenizer/context preflight precedes scanning | Focused reproducibility/HUMAN gate passes 42 tests; exact-head CPU gate passes 526 tests with 1 skip; independent re-review pending |
| 11 | re-review | `7917c9bd9ed27b0fa1d9bbbbf2a553792f9a5c87` against `origin/main` | Independent formal `/review` | `FAIL` | P1 in the inherited BENCH scanner: adding ordinary metadata fields let 124/128 selected JCommonsenseQA records evade complete contamination detection. The HUMAN cycle-9 repairs were not re-raised. | Reviewer independently reran 526 passed, 1 skipped and reproduced the enriched-record miss |
| 12 | target integration | `928b832dcaae747f0ad5a04644741b46ea526007` through `6414349fc147e40f6eaef86ca9041b3926468442` | Integrate repaired BENCH target `71c0eee929c1d5c37e47cb8cbc761f648e7630a9` | implemented | Adopted schema-owned mapping projections for augmented records plus nested-candidate follow-up; no duplicate local BENCH implementation | Enriched-record focused gate passes 2 tests; focused reproducibility/HUMAN gate remains 42 passed; exact-head CPU gate passes 531 tests with 1 skip; re-review pending |
| 13 | re-review | `9e47cdc5b5193377bf77bb1b1d1e95b07c3542c4` against BENCH target `71c0eee929c1d5c37e47cb8cbc761f648e7630a9` | Independent formal `/review` | interrupted; no verdict | The reviewer independently passed the 531-test suite, but the target advanced before a verdict; the stale-base review was stopped rather than represented as exact-head evidence | Reviewer output: 531 passed, 1 skipped; no verdict retained |
| 14 | target integration | `bd9924bfb9e25b096bc18a5e32d8e156426f1e5f` | Integrate BENCH target `f40d86333eb1b0136d429b0c27ce11856a571a45` | implemented | Adopted input-only projections for unlabeled JCommonsenseQA and GSM8K records; no HUMAN-local scanner fork | Exact-head CPU gate passes 532 tests with 1 skip plus Ruff, Hydra config, lock drift, and offline training/W&B smoke; re-review pending |
| 15 | re-review | `8dd94ee8e872e4686aefd76c0e06f9e0a33ed60c` against BENCH target `f40d86333eb1b0136d429b0c27ce11856a571a45` | Independent formal `/review` | `FAIL` | P2: the exclusive run-preparation marker survived process death and could permanently block checkpoint recovery | Reviewer independently reran 532 passed, 1 skipped and found no other issue |
| 16 | repair | `bd9331b766e3bf60c34b32932a5249d85afcfbb1` | Make preparation exclusion process-owned | implemented | Replaced the persistent `O_EXCL` marker with nonblocking advisory `flock`; an unlocked file left by process death is reusable while a live lock still fails closed | Focused reproducibility gate passes 17 tests; full CPU gate passes 533 tests with 1 skip plus Ruff, Hydra config, lock drift, and offline training/W&B smoke |
| 17 | target integration | `7d1aeae993935568d7b198a6e2d37cb6bf62cc9a` | Integrate BENCH target `c31e96ed9dff530388c663c26d2bed2c2ecdd6ee` | implemented | Adopted the formal `q_id` input-projection repair before certifying HUMAN | Exact integrated CPU gate passes 533 tests with 1 skip plus Ruff, Hydra config, lock drift, and offline training/W&B smoke; re-review pending |

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
- Change made: `6128ad2` rejects insufficient context before bundle publication
  and returns parsed score payload plus digest evidence from one read.
- Validation rerun: combined exact head `00c4045` passed 519 tests with one skip,
  plus Ruff, Hydra preflight, lock drift, and offline smoke.
- Remaining risk: cycle-3 independent review found three new integrity gaps.

### Review cycle 3 — formal re-review of `00c4045`

- Commit reviewed: `00c404521089d7694608ee55ec2b1ddaa985dea1`.
- Selected `CHECK.md` sections: applicable 7, 8, and 9.1; real DGX generation,
  human ratings, performance, and quality conclusions remained N/A.
- Ticket acceptance result: `FAIL`.
- Philosophy alignment: `FAIL` until cached clean evidence and reproducible
  blinding are bound to their real stable inputs.
- Complexity / change-surface result: no unnecessary abstraction finding.
- ML-system result: `FAIL`; stale contamination evidence could be reused and
  prompt/assignment identity could diverge from evaluated inputs.
- Verdict: `FAIL` because three actionable findings remained.

#### Findings

| Severity | Area | What was wrong | Required action |
| --- | --- | --- | --- |
| P1 | contamination cache | The HUMAN cache identity omitted transitive text producers, dependency lock, PyArrow, and Python/Unicode runtime | Bind the complete producer source, lock, and relevant runtime so stale clean evidence is ineligible |
| P2 | randomization | Evaluator operational config and volatile free-space evidence entered `study_id`, changing item order/A/B across workspace or cache moves | Derive assignment identity only from stable prompt/protocol/seed/checkpoint inputs; retain operational evidence privately |
| P2 | prompt identity | Prompt JSON was parsed before checkpoint loading and hashed by a later path read | Parse and hash one captured byte buffer |

## Failed-review handoff — cycle 3

- Reproduction command: `codex review --base origin/main` at `00c4045`.
- Constraints preserved: fixed balanced contract, exact checkpoint pair, private
  complete evaluator/scan evidence, no prompt text in scan reports, 100 GB free,
  no real generation/rating/GPU work.
- Repair request: close all three findings, add cache-invalidation,
  operational-identity, volatile-report, and prompt-replacement regressions,
  then rerun the exact-head CPU gate and independent review.

## Repair cycle 3

- Finding addressed: every cycle-3 finding.
- Change made: all repository Python producer bytes plus lock and relevant
  runtime now identity-bind prompt scans; study/bundle randomization excludes
  operational evidence; prompt parsing/hash share one byte buffer.
- Validation rerun: focused HUMAN gate passes 25 tests. Exact repair head
  `f92dfe9` passes 521 tests with one skip plus Ruff, Hydra preflight, lock
  drift, and offline smoke; 456 GB remained free.
- Remaining risk: cycle-4 independent review found two fail-closed ordering
  gaps outside the prior three repairs.

### Review cycle 4 — formal re-review of `f6ca50b`

- Commit reviewed: `f6ca50bf1e43ce4e68f7c237be78e43554c629f8`.
- Selected `CHECK.md` sections: applicable 7, 8, and 9.1; real DGX generation,
  human ratings, performance, and quality conclusions remained N/A.
- Ticket acceptance result: `FAIL`.
- Philosophy alignment: the evaluation design remained aligned, but fresh-run
  lineage and fail-fast corpus-work ordering were not yet safe enough.
- Complexity / change-surface result: no unnecessary abstraction finding.
- ML-system result: `FAIL`; a fresh launch could be mislabeled as a continuation,
  and known-unusable HUMAN requests could perform corpus-scale work.
- Verdict: `FAIL` because two actionable findings remained.

#### Findings

| Severity | Area | What was wrong | Required action |
| --- | --- | --- | --- |
| P1 | run lineage | A fresh launch into a directory with an existing manifest retained that manifest's `run_lineage_id`, including timestamp-collision cases | Reject occupied fresh launches; serialize same-directory preparation; retain lineage only for an explicit verified resume |
| P2 | HUMAN preflight | CPU/BF16, unavailable CUDA, unsupported CUDA BF16, and insufficient context were discovered only after a cache-miss corpus scan | Validate device, precision, tokenizer, and every prompt's fixed context before scanning |

## Failed-review handoff — cycle 4

- Reproduction command: `codex review --base origin/main` at `f6ca50b`.
- Constraints preserved: exact resume lineage, immutable run evidence, fixed
  HUMAN sampling, complete contamination scan for executable requests, 100 GB
  free, and no real generation/rating/GPU work.
- Repair request: fail closed before evidence mutation or corpus scanning, add
  occupied/concurrent-run and device/precision/context ordering regressions,
  then rerun the exact-head CPU gate and independent review.

## Repair cycle 4

- Finding addressed: both cycle-4 findings.
- Change made: `774d1bf` requires explicit verified resume lineage for an
  existing manifest, rejects fresh or concurrently colliding run preparation,
  and performs HUMAN device/precision/tokenizer/context checks before scanning.
- Validation rerun: focused reproducibility/HUMAN gate passes 42 tests. Exact
  repair/docs head `b6f03f9` passes 526 tests with one skip plus Ruff, Hydra
  config preflight, lock drift, and credential-isolated offline training/W&B
  smoke; 456 GB remained free.
- Remaining risk: cycle-5 review found one inherited BENCH integrity gap after
  the BENCH target had independently advanced with its repair.

### Review cycle 5 — formal re-review of `7917c9b`

- Commit reviewed: `7917c9bd9ed27b0fa1d9bbbbf2a553792f9a5c87`.
- Selected `CHECK.md` sections: applicable 7, 8, and 9.1 across the branch
  ancestry; real DGX generation, ratings, performance, and quality remained N/A.
- Ticket acceptance result: `FAIL` because the branch still carried an older
  inherited BENCH contamination scanner.
- Philosophy alignment: HUMAN-specific integrity repairs held, but benchmark
  isolation could not be certified on the stale target snapshot.
- Complexity / change-surface result: repair belongs in the BENCH target and is
  consumed here by target integration, not duplicated in HUMAN.
- ML-system result: `FAIL`; selected records enriched with ordinary metadata
  could receive reusable false-clean scan evidence.
- Verdict: `FAIL` because one P1 finding remained.

#### Finding

| Severity | Area | What was wrong | Required action |
| --- | --- | --- | --- |
| P1 | inherited BENCH contamination | Whole augmented mappings were hashed without also matching the selected record's schema-owned field subset; the review reproduced misses for 124/128 selected JCommonsenseQA examples | Integrate the advanced BENCH target containing enriched-record projection matching and its nested-candidate follow-up, then rerun validation/review |

## Failed-review handoff — cycle 5

- Reproduction command: `codex review --base origin/main` at `7917c9b`; the
  reviewer added one metadata key to each selected record and observed 124/128
  JCommonsenseQA misses.
- Constraints preserved: complete contamination detection, no weakening of
  scan work bounds, HUMAN stable identity and preflight repairs, 100 GB free,
  and no real generation/rating/GPU work.
- Repair request: consume BENCH target `71c0eee` (including `d9f6271` enriched
  record projections) rather than create a divergent HUMAN-local scanner fix.

## Repair cycle 5

- Finding addressed: the inherited BENCH P1.
- Change made: merge `928b832` integrates BENCH head `71c0eee`, including
  schema-owned field projections for augmented mappings and the nested
  candidate preservation follow-up.
- Validation rerun: the two enriched-record focused tests pass and the focused
  reproducibility/HUMAN gate remains 42 passed. Exact repair/docs head
  `6414349` passes 531 tests with one skip plus Ruff, Hydra config preflight,
  lock drift, and credential-isolated offline training/W&B smoke; 455 GB
  remained free.
- Remaining risk: independent exact-head re-review is pending.

## Interrupted re-review and advanced target integration

- The independent review of `9e47cdc` against BENCH target `71c0eee` was
  interrupted after its independent 531-test gate passed because the PR target
  advanced before the reviewer returned a verdict. It is retained as an
  interrupted review, not a pass.
- Merge `bd9924b` integrates BENCH head `f40d863`, whose contamination
  projections preserve detection for selected, unlabeled JCommonsenseQA and
  GSM8K input records.
- Exact `bd9924b` passes 532 tests with one skip plus Ruff, Hydra config
  preflight, lock drift, and credential-isolated offline training/W&B smoke;
  455 GB remained free. Independent exact-head review remains required.

### Review cycle 6 — formal re-review of `8dd94ee`

- Commit reviewed: `8dd94ee8e872e4686aefd76c0e06f9e0a33ed60c` against
  BENCH target `f40d86333eb1b0136d429b0c27ce11856a571a45`.
- Selected `CHECK.md` sections: applicable 7, 8, and 9.1; real DGX generation,
  human ratings, performance, and quality conclusions remained N/A.
- Ticket acceptance result: `FAIL` because interrupted training preparation
  could make an otherwise valid checkpoint recovery operationally inaccessible.
- Philosophy alignment: the lineage design held, but long-running recovery must
  not require undocumented manual lock-file repair after process death.
- Complexity / change-surface result: the repair is one standard OS advisory
  lock at the existing preparation boundary.
- ML-system result: `FAIL`; no model/data/evaluation finding, but recovery
  availability is part of the exact checkpoint contract.
- Verdict: `FAIL` because one P2 finding remained.

#### Finding

| Severity | Area | What was wrong | Required action |
| --- | --- | --- | --- |
| P2 | checkpoint recovery | The `O_EXCL` marker file remained after SIGKILL or host restart, causing every later preparation attempt to report a live collision | Use a process-owned advisory lock, or safely validate and reclaim stale ownership |

## Failed-review handoff — cycle 6

- Reproduction: leave `.run-preparation.lock` after the owner exits, then call
  `prepare_trainer` for a fresh or resume attempt in that run directory.
- Constraints preserved: reject simultaneous same-directory preparation,
  retain explicit verified resume lineage, fail before evidence mutation, keep
  the 100 GB reserve, and do no real GPU/rating work.
- Repair request: use a lock automatically released by the OS when the owning
  process dies and add both live-owner rejection and stale-file reuse coverage.

## Repair cycle 6 and final target integration

- Finding addressed: interrupted preparation no longer leaves stale ownership.
- Change made: `bd9331b` uses nonblocking `flock` on the stable preparation-lock
  inode. A live owner remains exclusive; process death closes the descriptor
  and releases ownership without relying on a `finally` block or file removal.
- Validation rerun: focused reproducibility gate passes 17 tests; exact repair
  head passes 533 tests with one skip plus Ruff, Hydra config preflight, lock
  drift, and credential-isolated offline training/W&B smoke.
- Target integration: merge `7d1aeae` consumes BENCH head `c31e96e`, including
  its formal `q_id` input-projection repair. The exact integrated head again
  passes 533 tests with one skip plus the complete CPU gate; 455 GB remained
  free. Independent exact-head re-review remains required.

## Independent re-review

- Commit reviewed: pending cycle-17 target-integration successor.
- Prior findings disposition: cycle-1 through cycle-6 repairs implemented;
  exact-head re-review pending.
- New findings: pending.
- Verdict: pending.
- Evidence: pending.

## Merge authority and guarded audit

- Merge path: human review/merge by default; no merge is authorized from this
  incomplete review state.
- Human authorization and scope, or `N/A — human merge`: bounded roadmap series
  authorization exists, but review and exact-head gates still block merge.
- Exact reviewed head: latest reviewed head `8dd94ee`; failing.
- Final review verdict: `FAIL`.
- Actionable findings repaired and independently re-reviewed: no.
- Blocking review decision / newer human objection: actionable review findings
  remain until the successor passes independent re-review.
- Unresolved review threads: PR audit pending.
- Required-context and configured-workflow inventory: pending final audit.
- Exact-head check statuses: integrated target head `7d1aeae` passed 533 tests
  with one skip plus Ruff, Hydra config, lock drift, and offline
  training/W&B smoke; independent review pending.
- Current base and mergeable evidence: BENCH target `c31e96e` is integrated;
  final GitHub refresh remains pending.
- PR trail, validation, risks, and authorization parity: this record and PR #50
  must retain all failed cycles.
- Prohibited self-merge categories: no prohibited external action is requested,
  but failed review independently blocks merge.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: PR #50, pending.
- Immediate pre-merge refresh location and drift result: pending.
- Merge outcome: not merged.
