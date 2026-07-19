# <TICKET> — <One-Sentence Hypothesis>

- Roadmap ticket:
- Branch: `codex/<ticket-lower>-<slug>`
- Draft PR: pending / URL / unavailable with reason
- Experiment owner:
- Status: planned / running / concluded
- Started (UTC):
- Last updated (UTC):
- Review evidence: draft PR body/comments

## Predeclared question and decision rule

- Hypothesis (one falsifiable claim):
- Expected result:
- Success condition:
- Failure condition / stop condition:
- Relevant baseline commit and run:
- Baseline metrics and evidence link:
- Smallest run capable of answering the question:

Do not edit the predeclared conditions after the first attempt. Add a dated note
explaining a correction and apply it only to a new attempt.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware |  |  |
| Training tokens |  |  |
| Optimizer steps |  |  |
| Evaluation work and cadence |  |  |
| Checkpoint count and bytes |  |  |
| Local / external / W&B storage |  |  |

## Attempt <N> — <short outcome>

Never delete an attempt. Copy this section for every command that could have
produced decision-relevant evidence, including failed, aborted, and negative
runs.

### Launch identity

- Started / ended (UTC):
- Outcome: succeeded / negative / failed / aborted
- Exact command:
  ```text
  <verbatim command>
  ```
- Fully resolved Hydra configuration or immutable path to it:
  ```yaml
  <complete composed configuration, not only overrides>
  ```
- Git commit SHA:
- Worktree state: clean / dirty; if dirty, immutable diff or patch path:
- Dependency lock identity (path and checksum):
- Container/image identity or `N/A` with reason:

### Scientific identity

- Model architecture/config identity and parameter count:
- Initialization / pretrained-weight check:
- Tokenizer name, immutable revision/checksum, and special-token contract:
- Training manifest identity/checksum:
- Validation manifest identity/checksum:
- Train/validation disjointness evidence:
- Random seeds (Python, NumPy, PyTorch CPU/CUDA, loader/sampler as applicable):
- Hardware identity:
- Software/runtime identity:
- Precision and numerical controls:

### Counters, evidence, and integrity

- Actual elapsed time, optimizer steps, training tokens, examples, and target
  tokens:
- Training/validation/evaluation metrics with denominators:
- System metrics (throughput, memory, utilization, I/O, pauses as applicable):
- W&B run/project/artifact IDs, or reasoned `N/A`:
- Checkpoint IDs/paths/checksums and retention status, or reasoned `N/A`:
- Logs, tables, plots, or other immutable evidence links:
- Integrity checks (data identity and separation, no benchmark leakage, no
  pretrained weights/teacher/synthetic targets, finite values, counter
  consistency, checkpoint/config agreement as applicable):
- Comparison with the predeclared baseline using the same protocol:

### Attempt interpretation

- Result against success/failure conditions:
- Failure or anomaly:
- Most likely cause and supporting evidence:
- Alternatives ruled out and how:
- What remains uncertain:

## Conclusion

- Hypothesis result: supported / not supported / inconclusive
- Evidence-backed conclusion:
- Uncertainty and limitations:
- Exactly one next step or next question:
