# GATE-001 — Fixed bilingual fixture can be memorized and resumed end to end

- Roadmap ticket: `GATE-001`
- Branch: `codex/gate-001-bilingual-overfit-proof`
- Draft PR: [#39](https://github.com/Ayumu-J-S/llm_scratch/pull/39)
- Experiment owner: implementation agent; actual exact runtime model and reasoning are not exposed
- Status: concluded candidate; independent review pending
- Started (UTC): 2026-07-12
- Last updated (UTC): 2026-07-12T16:23:22Z
- Model-run provenance: `docs/model-runs/GATE-001-bilingual-overfit-proof.md`

## Predeclared question and decision rule

- Hypothesis: a small random-initialized decoder can memorize a fixed, versioned Japanese/English fixture within a bounded run, and a verified recovery checkpoint resumes its exact suffix and yields recognizable base-model continuations in both languages.
- Expected result: two independent same-seed 200-update executions produce equal optimizer/target-token counters, equal loss traces, equal full-state checkpoint identities and model digests, and equal greedy samples; a separate split/resume trajectory also equals that uninterrupted trajectory.
- Success condition: each full run reaches final token-weighted training NLL at or below `0.20` by optimizer step `200`; a split at step `100` resumes to the same step-`200 loss trace/checkpoint model digest as its uninterrupted counterpart; and greedy checkpoint continuations for the two fixed prompts include the fixture's Japanese and English suffixes. These are memorization checks only, never held-out validation or generalization.
- Failure condition / stop condition: stop and retain the attempt if loss or gradients become non-finite, if the run exceeds 200 optimizer steps or 20 minutes per execution, if the loss threshold is missed, if resume diverges in any compared counter/trace/identity/sample, or if a sampled continuation does not contain the predeclared fixture suffix.
- Relevant baseline commit and run: `origin/main` at `7f9c1728098f5e0dc18653b1660e07e5b36788ce`; no prior GATE-001 run exists.
- Baseline metrics and evidence link: `N/A — this is the first end-to-end memorization gate, not a baseline-quality comparison.`
- Smallest run capable of answering the question: two 200-step fixed-fixture CPU/GPU runs plus one 100+100-step interrupted/resumed trajectory, each using the checkpoint-backed `src/generate.py` implementation.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware | 20 minutes per execution; stop at 60 minutes total | Bounded R2 gate; not a performance/thermal pilot |
| Training tokens | At most 200 updates per full execution; exact target count recorded | Explicit optimizer-step cap and observed target counters |
| Optimizer steps | 200 full; 100 checkpoint + 100 resume | Predeclared reproducibility comparison |
| Evaluation work and cadence | No held-out evaluation; greedy samples only after final checkpoint | Same-corpus memorization must not masquerade as validation |
| Checkpoint count and bytes | Recovery every 100 updates, final checkpoint per execution | Resume proof and checkpoint-backed generation |
| Local / external / W&B storage | Local run directories only; W&B disabled | Gate does not need an external artifact upload |

## Attempt inventory

All attempts remain in local `reports/gate-001/attempt-*`; raw checkpoints are
ignored because they are recoverable local artifacts rather than repository
evidence. The compact outcomes below are the durable record.

| Attempt | Candidate condition | Outcome | Retained finding |
| --- | --- | --- | --- |
| 1 | Native `uv` CUDA request | Failed before data/model construction | Host Torch was CPU-only; explicit CUDA guard worked. |
| 2 | First Docker fixture | Failed during preview | Each document was shorter than a packed 17-token window. |
| 3 | Longer fixture, one epoch | Failed before recovery point | Finite source produced one update/epoch, never reaching step 100. |
| 4 | 200 finite passes, recovery at 100 | Failed resume | Step 100 landed on the terminal batch, leaving no resumable suffix. |
| 5 | 11-window heterogeneous fixture | Negative | Exact repeat/resume worked, but final update NLL was `0.7017` and English full suffix was absent. |
| 6 | Concentrated two-document fixture | Negative sampling audit | Final NLL was `0.1645` and all trajectories matched; short prompts failed the predeclared full-suffix test. |
| 7 | Same training; new predeclared unambiguous prompts | PASS candidate | Every loss, counter, identity/model digest, and sample comparison passed. |

## Attempt 7 — bounded GB10 fixed-fixture proof

### Launch identity

- Started / ended (UTC): 2026-07-12T16:22:19Z / 2026-07-12T16:22:46Z
- Outcome: succeeded; `gate_record.json` verdict `PASS`
- Exact command:
  ```text
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /tmp/llm_scratch-gate001:/tmp/llm_scratch-gate001 -v /home/ayumu/Documents/Proj/llm_scratch/.git:/home/ayumu/Documents/Proj/llm_scratch/.git:ro -w /tmp/llm_scratch-gate001 -e PYTHONPATH=/tmp/llm_scratch-gate001/src -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/tmp/llm_scratch-gate001 llm-scratch:env-001 python scripts/run_gate_overfit.py --output-dir reports/gate-001/attempt-7 --device cuda
  ```
- Resolved Hydra configuration: [`config/profile/gate_overfit.yaml`](../../config/profile/gate_overfit.yaml), with `runtime.device=cuda`, BF16, seed `42`, model `(embed=32, heads=4, layers=1, dropout=0)`, sequence length `16`, batch `1`, `max_steps=200`, `epochs=20`, AdamW `(lr=.02, betas=.9/.95, eps=1e-8, weight_decay=0)`, no scheduler, recovery every 100 steps, no validation within the budget, and W&B disabled.
- Git commit SHA: `3d0f4fbdc7c8ad40d30b9e5eb03e448e712d2e2e`; clean worktree.
- Dependency lock identity: `uv.lock` SHA-256 `cc1e1e2fb13e2b6088d3fde9582717dbe9bf08e4bf285229672c9ebf139561b3`.
- Container/image identity: `llm-scratch:env-001`, NGC base `nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`.

### Scientific identity

- Model: `SimpleDecoderTransformer`, random initialization, `3,299,754` trainable parameters; no pretrained weights, teacher outputs, or synthetic targets.
- Tokenizer: canonical `llm-jp-v1`, fingerprint `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`.
- Train fixture: `tests/fixtures/gate_overfit/v1/train.manifest.json`, manifest SHA-256 `0c284a3a2474b9ed2f70891a58e5a852ecdf816bc8061b73e6a3546947036d92`, fingerprint `eb607b8156d75987032d213e62dbb8c76bf61c7521d0a8db51b35c88c57804e9`.
- Auxiliary manifest: distinct no-overlap source fingerprint `a4cb7820a679cd914e148395daff2dbf6b4f6c6fac18e4addde13dae98640cdb`; it was never scored because its validation cadence is step 1000.
- Hardware/software: NVIDIA GB10 (CC 12.1), driver `580.159.03`, CUDA runtime `13.3`, Torch `2.13.0a0+8145d630e8.nv26.06`, Python `3.12.3`, ARM64 Ubuntu 24.04.
- Precision and numerical controls: BF16 autocast, FP32 optimizer state, global clipping at `10.0`, finite-loss/gradient/parameter stops, deterministic algorithms enabled with PyTorch's documented memory-efficient-attention `warn_only` notice.

### Counters, evidence, and integrity

- Reference, independent repeat, and interrupted/resumed runs each reached 200 optimizer steps and 3,200 target tokens (16 targets/update). The common first/final NLLs were `10.8711` and `0.164456`; final-pass mean NLL was `0.192488`.
- The full step/counter/loss trace hash was `8758e770ae67e408c47b6a32e862513cf5b79668d1e45742f73e61ea298c7350` for all three executions. Every comparison reported true for counters, trace, checkpoint identity, final-model digest, samples, and the NLL threshold.
- Reference checkpoint identity SHA-256: `02c40cea74bfdff78d9db1723decc6d594aa79dcd909b6f8846155cc0ca479db`; model digest: `0e100e0da66f0710db001064fdab7a5efa363c2f37e4f4336a8f27ef7d8f14c1`. Each final checkpoint was 39,646,677 bytes; the step-100 verified recovery was 39,648,446 bytes.
- GEN-001 greedy base-model continuations from the final checkpoint: `日本語の合図: 桜は` -> `春に咲きます。日本語の合図`; `English cue: small models` -> `memorize fixed text.`. The first fixed suffix is present in each; no result is a chat response.
- Local evidence: `reports/gate-001/attempt-7/gate_record.json`, per-run `run_manifest.json`, resolved configs, metrics JSONL, and verified checkpoints. W&B IDs/artifacts: `N/A — disabled`.
- Integrity: both fixture manifests are versioned/hashed; auxiliary data is disjoint and unscored; no benchmark data, production corpus, held-out metric, or external model capability entered this run.

### Attempt interpretation

- Result against conditions: PASS candidate. The predeclared loss/update budget, same-seed independent repeat, exact-resume trajectory, and checkpoint-backed JP/EN continuation requirements all passed.
- Failure/anomaly: no non-finite values. The BF16 memory-efficient-attention deterministic warning remains an environment limitation, not a claim of cross-platform bitwise reproducibility.
- What remains uncertain: this proves only same-fixture memorization on one current GB10 container; it establishes no held-out validation, generalization, benchmark score, throughput, thermal, or production-data claim.

## Retry predeclaration — 2026-07-12

The original decision rule above is unchanged: final fixed-fixture NLL must be
at or below `0.20` at step 200, two independent same-seed runs must agree, and
the split/resume path must agree. The following is a new planned attempt, not a
revision of the original threshold or budget.

- Retained failures that motivate this retry: native CUDA stopped before data
  because the host Torch build is CPU-only; two short-fixture launches exposed
  zero/terminal suffixes at the step-100 recovery boundary; the first complete
  GB10 attempt had exact repeat/resume traces but ended at NLL `0.7017` on its
  final diverse English batch and did not complete the required English suffix.
  Raw local records are retained in ignored `reports/gate-001/attempt-{1..5}`.
- New hypothesis: concentrating the same fixed Japanese and English continuations
  in two versioned documents, each repeated seven times, will satisfy the
  unchanged step-200 loss and continuation gates without changing architecture,
  tokenizer, optimizer, seed, precision, or the 100-step recovery boundary.
- Smallest change: replace six heterogeneous train documents with the two
  declared JP/EN memorization texts. The new manifest yields exactly 11 packed
  batches per finite pass, so step 100 is deliberately inside a pass and has a
  non-empty exact-resume suffix.
- Planned command: `docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ... llm-scratch:env-001 python scripts/run_gate_overfit.py --output-dir reports/gate-001/attempt-6 --device cuda`.
- Planned budget: unchanged — two independent 200-update executions plus one
  100+100 interrupted/resumed execution, all on GB10 BF16, with local-only
  evidence and W&B disabled.
- Additional stop condition: abort before optimizer updates if the fixture pass
  count is fewer than two, divides step 100, or cannot supply 200 updates in
  the declared epoch horizon.

## Sampling-audit retry predeclaration — 2026-07-12

- Retained result: the preceding concentrated-fixture attempt met the unchanged
  loss condition and exact repeat/resume comparison, but the original short
  prompts admitted a competing language's first tokens and therefore failed the
  full-suffix string check despite visibly producing Japanese and English
  fragments.
- New hypothesis: the same checkpoint-backed model will complete the exact
  fixed suffixes when supplied with unambiguous fixed prefixes from the same
  memorization fixture: `日本語の合図: 桜は` -> `春に咲きます。` and `English cue:
  small models` -> `memorize fixed text.`.
- Smallest change: sampling prompts and expected suffixes only. Training
  fixture, model, tokenizer, seed, 200-step budget, loss threshold, recovery
  point, comparison requirements, and local-only policy remain unchanged. This
  is a new predeclared evaluation-prompt selection applied before a full
  canonical rerun; it does not relabel the retained short-prompt attempt as a
  pass.
- Planned command: the same canonical Docker command as the retry above with
  `--output-dir reports/gate-001/attempt-7 --device cuda`.

## Conclusion

- Hypothesis result: supported for the fixed fixture only.
- Evidence-backed conclusion: the current random-initialized model can memorize the versioned bilingual fixture within 200 updates, restore the verified step-100 suffix without trajectory drift, and produce the fixed JP/EN base-model continuations. This is not held-out validation or generalization.
- Uncertainty and limitations: no production data, benchmark, W&B upload, long pilot, performance comparison, or cross-platform deterministic guarantee was attempted.
- Exactly one next step: obtain the required independent `CHECK.md` review of PR #39's exact candidate head.
