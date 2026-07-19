# Blinded base-model continuation evaluation

HUMAN-001 compares two checkpoints from one RUN-001 pretraining run. It is a
small qualitative continuation study, not preference-data collection and not a
chat evaluation. The checked-in `HUMAN-001-v1` prompt set contains four
Japanese and four English prompts. Each checkpoint receives the same derived
seed for a prompt under the fixed generation contract:

- `max_new_tokens=64`
- `temperature=0.8`
- `top_k=40`
- one explicit master seed, with a private HMAC-derived per-prompt seed

The two full-state checkpoints must be physically distinct, reconstructable by
the canonical sampler, belong to the same unique `run_lineage_id` as well as the
same experiment/config/tokenizer/data identity, advance both optimizer-step and target-token counters, and be at
least 25% apart using the later checkpoint's target-token count as the
denominator. The real Hydra workflow defaults to `device=cuda` for canonical
BF16 RUN-001 checkpoints. Sampling uses the checkpoint-owned `training.precision`
and the exact physical SHA-256 verified for the private mapping. It loads and
finishes one checkpoint before loading the other, so it does not keep two
models resident on the DGX Spark.

Before either model is loaded for generation, the workflow scans every document
in every checkpoint-owned training manifest for exact and normalized occurrences
of all eight prompts. Any occurrence blocks generation and leaves an
authenticated private report containing prompt IDs, source/document IDs,
counts, and digests but no prompt text. The scan cache is identity-bound to the
prompt set, run/config/data manifests, and scanner implementation, and enforces
at least 100 GB free disk. Generation uses the same strict deterministic CUDA
policy as BENCH-001: deterministic algorithms, math SDPA only, TF32 disabled,
and a pre-initialization cuBLAS workspace policy.

## Prepare the study

Keep the HMAC key outside the evaluation workspace and outside the repository.
The workflow anchors that check to the evaluator checkout itself and accepts
prompt assets only from that checkout's `evaluation/human` directory; overriding
the prompt path cannot redefine which repository is protected.
The command creates it once with exact owner-only `0600` permissions and refuses
to replace it:

```bash
uv run llm-scratch-human-evaluate \
  action=create_key \
  blinding_key_path=/home/USER/.config/llm-scratch/secrets/HUMAN-001.key
```

Prepare a new dedicated workspace. Generated workspaces are ignored by Git and
must include a `human-evaluation` path component. Paths that overlap `data`,
`runs`, `checkpoints`, `cache`, `manifests`, `artifacts`, or training
namespaces are rejected.

```bash
uv run llm-scratch-human-evaluate \
  action=prepare \
  workspace_dir=/absolute/operator/path/human-evaluation/RUN-001-milestones \
  blinding_key_path=/home/USER/.config/llm-scratch/secrets/HUMAN-001.key \
  checkpoints.earlier=/absolute/run/path/checkpoints/milestone-EARLIER.pt \
  checkpoints.later=/absolute/run/path/checkpoints/milestone-LATER.pt \
  generation.seed=20260719 \
  device=cuda
```

The opaque study ID is an HMAC over the versioned prompt set, generation seed,
exact private checkpoint pair, completed contamination report, deterministic
policy, and evaluator identity. The assignment is reproducible with the same
key and inputs. Within each language, the earlier checkpoint appears as A twice
and B twice, preventing checkpoint position from being confounded with language.

The workspace separates material by purpose:

```text
human-evaluation/RUN-001-milestones/
  public/bundle.json       # only file shared with reviewers
  private/mapping.json     # authenticated checkpoint mapping, mode 0600
  reviews/                 # returned score files, directory mode 0700
  private/result-*.json    # unblinded result after import, mode 0600
```

Only `public/bundle.json` leaves the operator's private workspace. It contains
opaque study/bundle/item IDs, prompt language/text, literal A/B continuations,
the protocol/evaluator revision and device, the fixed non-seed sampling
settings, and the rubric. The opaque bundle ID is HMAC-bound to the exact
continuations, device, checkpoint-owned precision, and private generation seed;
score files cannot cross-apply to another bundle even when its study/item IDs
match. The closed public schema has no checkpoint, run, path, checkpoint hash,
counter, seed, ordering, or mapping fields. The
private mapping records exact checkpoint paths/SHA-256/counters and generation
seeds, the unique run lineage, complete clean contamination report, deterministic
policy, and evaluator identity. The evaluator identity includes Git HEAD plus
dirty-content digest, `uv.lock`, complete resolved HUMAN Hydra configuration,
and OS/Python/PyTorch/CUDA/container details. HMAC-SHA256 authenticates this
private evidence and binds the public bundle hash.

## Human scoring

Use at least two genuinely distinct human reviewers. Do not prefill, simulate,
or model-generate ratings. A reviewer scores each candidate independently on
the public 1–5 rubric and chooses `A`, `B`, or `tie`. Save one returned JSON file
per reviewer under the study's private `reviews/` directory and make both the
directory and files private:

```bash
mkdir -p -m 700 /absolute/operator/path/human-evaluation/RUN-001-milestones/reviews
chmod 700 /absolute/operator/path/human-evaluation/RUN-001-milestones/reviews
chmod 600 /absolute/operator/path/human-evaluation/RUN-001-milestones/reviews/*.json
```

Each score file uses this exact schema, with one rating for each of the eight
opaque item IDs in the public bundle:

```json
{
  "schema_version": "human-evaluation-scores-v1",
  "study_id": "study-OPAQUE",
  "bundle_id": "bundle-OPAQUE",
  "reviewer_id": "distinct-private-reviewer-id",
  "ratings": [
    {
      "item_id": "item-OPAQUE",
      "candidate_a": {"fluency": 1, "coherence": 1, "naturalness": 1},
      "candidate_b": {"fluency": 1, "coherence": 1, "naturalness": 1},
      "preference": "tie",
      "comment": ""
    }
  ]
}
```

The example values show the schema, not ratings to copy. Reviewers must replace
them with their own judgments. Import is fail-closed: it rejects extra or
missing fields, missing/duplicate items, out-of-range scores, a wrong study,
a wrong exact bundle ID, case-insensitively duplicate reviewer IDs, fewer than
two reviewers, wrong file modes, public-bundle changes, private-mapping changes,
or the wrong key.

```bash
uv run llm-scratch-human-evaluate \
  action=import_scores \
  workspace_dir=/absolute/operator/path/human-evaluation/RUN-001-milestones \
  blinding_key_path=/home/USER/.config/llm-scratch/secrets/HUMAN-001.key \
  'scores.paths=[/absolute/operator/path/human-evaluation/RUN-001-milestones/reviews/reviewer-1.json,/absolute/operator/path/human-evaluation/RUN-001-milestones/reviews/reviewer-2.json]'
```

The private result retains each score-file checksum, unblinds every rating to
the exact earlier/later checkpoint identity, reports per-checkpoint rubric
means and preferences, and computes reviewer-pair denominators, exact
preference agreement, exact rating agreement, and rating mean absolute
difference. It also retains preparation/import evaluator identities, the fixed
determinism policy, and the complete prompt-contamination report. A new
score-file set receives a new immutable result filename.

## Research-integrity boundary

The prompt asset stays under `evaluation/human`, and generated prompts,
continuations, mappings, scores, and results stay under the ignored dedicated
`human-evaluation` namespace. None of this material is a training example,
preference dataset, manifest input, cache entry, benchmark input, or target.
Never copy it into those paths. HUMAN-001 remains incomplete until RUN-001
provides the real separated checkpoints and at least two actual humans complete
the blinded bundle.
