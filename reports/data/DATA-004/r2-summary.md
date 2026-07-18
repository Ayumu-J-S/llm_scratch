# DATA-004 real-data R2 target smoke

- Status: `PASS`
- Code: `fee0f1a231e24957cee86568d9ef89f04eb4e27d`
- Runtime: pinned `llm-scratch:env-001` image on NVIDIA GB10, CUDA/BF16
- Resolved config: `r2-resolved-config.yaml`
- Run identity: `r2-run-manifest.json`
- Raw scalar evidence: `r2-metrics.jsonl`

The host uv environment intentionally failed before data loading because its
PyTorch build has no CUDA. The same command then ran through the repository's
digest-pinned DGX image; the profile was not weakened to CPU.

## Result

| Measure | Observation |
| --- | ---: |
| Optimizer steps | 50 |
| Trained targets | 3,200 |
| Time through final update | 5.921 s |
| Target tokens/s | 540.44 |
| Step time median / p95 | 93.10 / 95.92 ms |
| First / last step loss | 11.0293 / 9.9233 |
| Gradient norm min / max | 4.2006 / 12.7288 |
| Clipped updates | 5 |
| Non-finite values | 0 |
| Validation loss | 9.67945 |
| Container max RSS | 709,120 KiB |
| Temperature range | 36–41 °C |
| Sampled power range | 4.45–19.45 W |
| Sampled SM clock range | 208–2,411 MHz |
| Sampled utilization range | 0–68% |

All three best/recovery/final checkpoints verified. Each is about 594.6 MB;
the observed local checkpoint footprint was 1,783,745,333 bytes. The run used
the exact Japanese, English, and tokenizer fingerprints recorded in the JSON
summary.

This is an R2 correctness/stability smoke, not a model-quality result or the
DGX-001 model-sizing decision. The known memory-efficient-attention
non-determinism warning remains recorded.
