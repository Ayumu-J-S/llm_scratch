# Philosophy

## Mission

This repository exists to train a small, capable autoregressive language model
from random initialization and to make the process understandable from end to
end.

It is both a model project and an experimental system. The model should be
useful, but the path to it matters just as much: a person should be able to read
the code, inspect a run, understand why a decision was made, and improve the
system without first untangling an opaque research prototype.

The project is also a learning instrument for people. It should make the full
causal chain visible: text becomes tokens and batches; the model produces
logits; the objective produces loss and gradients; optimization changes the
weights; checkpoints improve over time; and those changes eventually become
observable language behavior. Someone studying the repository should be able to
connect a generated output to the training process that produced it.

The current stage is pretraining. Evaluation is part of pretraining from the
beginning. Supervised fine-tuning may come later, but it is not a present goal.
Inference exists only to support correctness checks, sampling, and evaluation;
optimizing or productizing inference is out of scope.

## One machine is the boundary

The complete training workflow must fit on one DGX Spark. The project may use
smaller environments for development and smoke tests, but the real model must
not depend on a multi-node cluster or a hidden external training service.

The hardware limit is a design constraint, not an excuse to hide complexity.
Model size, context length, batch construction, precision, checkpointing, and
evaluation cadence should be chosen together so that a full run is practical on
the target machine. We prefer a smaller experiment that completes and teaches
us something over a larger one that is operationally fragile.

## Begin with a conventional model

The first credible baseline should be intentionally ordinary: token embeddings,
a simple positional representation, conventional causal self-attention,
feed-forward layers, residual connections, normalization, and a language-model
head. These components should be implemented clearly enough that their behavior
can be inspected and tested.

We do not begin with exotic attention mechanisms, mixture-of-experts routing,
retrieval, or other architectural novelty. First establish that a plain
decoder-only Transformer can overfit a tiny sample, train stably on the real data
pipeline, and produce sensible validation behavior. A more complex mechanism is
introduced later only as a focused experiment with a measurable reason to
exist.

Simple architecture does not mean careless software. The baseline is the
reference against which later experiments are judged, so it should be the
cleanest and most trustworthy path in the repository.

The baseline is not complete merely because the code runs. It should be
pretrained far enough to produce recognizable, increasingly coherent Japanese
and English continuations, so a person can see that the learning process has
created real behavior. During the pretraining stage, a "response" means a
base-model continuation or prompt completion, not instruction following or
chat-assistant behavior. That distinction should remain visible in examples and
evaluation.

Starting conventionally is a sequencing principle, not a permanent ban on new
ideas. Once the baseline works and is understood, architectural and training
variations become easier to isolate, explain, and judge honestly.

## Train the model we claim to train

Model weights begin from random initialization. This repository is for creating
a model through our own experiments, not for repackaging capability learned
elsewhere.

We do not use:

- pretrained model weights or continued pretraining;
- teacher logits or knowledge distillation;
- model-generated synthetic training data;
- outputs from another model as training targets; or
- a remote or embedded model as a hidden part of the final system.

Existing models may be evaluated as external baselines, under the same published
protocol, but their weights and outputs must not enter our training pipeline. An
existing tokenizer is allowed because tokenizer selection is an explicit early
project decision, not a substitute for learned model weights.

Training data must be kept separate from benchmark test data. Data provenance,
dataset versions, split construction, and known contamination risks must be
recorded. We do not tune on a held-out test set, cherry-pick favorable runs, or
omit failed experiments from the research record.

## Japanese and English first

Pretraining data should emphasize Japanese and English. Useful data in other
languages is welcome; this is not a requirement to build a strictly bilingual
model. The default objective is ordinary left-to-right next-token prediction
over a documented data mixture.

We initially adopt one established tokenizer with strong Japanese support rather
than making tokenizer research a prerequisite for model research. The choice
should be justified by license, vocabulary, Japanese and English compression,
runtime behavior, and integration quality. Tokenization remains a clean
component boundary so it can become an experiment later if measurements show
that it is limiting quality or throughput.

The same rule applies to data. We begin with a credible, reproducible mixture,
measure its behavior, and replace or rebalance it when evidence identifies a
bottleneck. A dataset is not permanent merely because it was used first.

## Evaluation is part of training

Training loss alone is not enough. We want to see when general language ability,
Japanese ability, and reasoning begin to emerge as pretraining progresses.
Evaluation therefore runs on intermediate checkpoints, not only on the final
model.

Evaluation has separate cadences:

1. Lightweight validation runs frequently on fixed held-out corpora.
2. Broader benchmark evaluation runs less frequently on checkpoint milestones.
3. Human evaluation runs on selected milestones where qualitative comparison is
   worth the review cost.

Cadence is configured in optimizer steps or tokens seen. Epoch-based cadence may
be used for a bounded dataset, but it is not the primary unit because an epoch
can become ambiguous when corpora are streamed, mixed, or changed. Evaluation
and checkpoint intervals belong in Hydra configuration and must be adjustable
without editing source code.

The evaluation suite should include Japanese language tasks and general ability
tasks, including mathematical and multi-step reasoning tasks such as GSM8K. A
development subset may be used repeatedly during training, while final test sets
remain reserved for infrequent, decision-grade evaluation. Prompts, few-shot
examples, decoding settings, dataset revisions, and scoring code are versioned
so scores remain reproducible.

Small open models may be included as comparison baselines. Comparisons must use
the same evaluation implementation and must disclose meaningful differences in
parameter count, training compute, tokenizer, context length, and data access.
The purpose is to understand our model's level, not to manufacture a favorable
ranking.

Human evaluation should be structured and, where practical, blinded. For a base
model it should evaluate completions appropriate to a pretrained model rather
than penalize the model for not yet behaving like an instruction-tuned
assistant.

## Experiments are first-class software artifacts

Each pretraining experiment begins with one explicit hypothesis and lives on a
focused Git branch. The stable branch represents the best reproducible baseline;
an experiment branch should make the smallest coherent change needed to test its
hypothesis. Clean component boundaries should let most experiments replace or
modify one part of the system without rewriting the rest.

Compute budgets are planned primarily in elapsed time on the DGX Spark, then
translated into tokens, steps, evaluation work, checkpoints, and storage using
measured throughput. Before a substantial run, the agent estimates those costs
and chooses the smallest run capable of answering the question. The estimate is
a planning tool, not a reason to wait for permission.

Experiments proceed as an evidence-driven loop: implement the smallest coherent
change, run a smoke test, launch the planned run, inspect the result, and adjust
the implementation or configuration when the evidence shows a problem. A failed
attempt should normally lead to diagnosis and a documented retry, not an
immediate handoff to the human.

Every consequential run records:

- the question and expected outcome;
- the Git commit and complete Hydra configuration;
- model, tokenizer, and dataset versions;
- random seeds and target hardware;
- training, validation, evaluation, and systems metrics;
- checkpoint and Weights & Biases artifact identifiers;
- comparison with the relevant baseline; and
- a conclusion, including failures and unresolved uncertainty.

Weights & Biases is the experiment system of record. At minimum, runs should make
loss, learning rate, tokens seen, throughput, memory behavior, validation
metrics, benchmark results, and relevant stability signals visible. Logging is
not a substitute for interpretation: the branch or pull request must contain a
short explanation of what the evidence means.

The Weights & Biases account is expected to operate within the Free plan, so W&B
is not our bulk storage layer. Metrics, configuration, compact tables, summaries,
and lineage are high-value uploads; repeated copies of corpora and checkpoints
are not. Dataset contents stay on controlled local or external storage by
default. We track them with versioned manifests, checksums, provenance, and,
where useful, reference artifacts that record metadata without uploading the
underlying files.

Checkpoint retention balances recovery cost against storage cost. Training keeps
a small rotating set of local recovery checkpoints. W&B model artifacts are
reserved for models small enough to fit the current quota and for checkpoints
with a reason to persist: normally the best, final, or an explicitly selected
milestone model, not every checkpoint. Save cadence is independent of evaluation
cadence, configured in steps or tokens, and should not be more frequent than the
expected cost of recomputing lost work justifies. Old local recovery checkpoints
may be removed only after a newer checkpoint has been verified.

Agents check the current W&B plan, usage, and retention behavior before a bulk
upload. Storage limits can change and therefore do not belong as hard-coded
constants in the codebase. Artifact TTL may be used when available, but deletion
is not assumed to release quota immediately. The default strategy is to avoid
unnecessary uploads in the first place.

A positive result from one noisy run is a lead, not a conclusion. The amount of
replication should be proportional to the cost and importance of the decision.

## Agent-native, human-legible

An agent should be able to read this repository, identify the current baseline
and the next unanswered question, plan a bounded experiment, implement it, run
checks, launch training, monitor Weights & Biases, analyze the result,
and prepare the next decision. The human should not need to decompose every
routine step. Agents may use subagents when that makes the work faster or more
reliable.

This document is the default decision policy when no narrower instruction exists.
An agent uses it to choose and prioritize the next useful work: protect research
integrity, establish the simple baseline, maximize learning per unit of time,
produce reproducible evidence, and leave a reviewable handoff. It should resolve
ordinary implementation choices without turning each one into a question for the
human.

Safe, reversible work within this repository and on its DGX Spark has standing
authorization. This includes editing code and configuration, managing experiment
branches, running tests and benchmarks, launching short or long training runs,
stopping invalid runs, changing a failed run's configuration, and retrying it.
Agents should exercise judgment and proceed without repeatedly asking whether
ordinary in-scope work is allowed.

Autonomy includes responsibility. An agent must:

- validate the data path and run a small smoke test before consuming substantial
  compute;
- investigate unknown technical or operational facts using current official
  documentation, primary sources, and local measurements before asking the
  human, and record sources that materially affect a decision;
- monitor long runs and stop or repair runs that are clearly invalid;
- preserve failed-run evidence and explain deviations from the plan;
- ask a human when credentials such as Weights & Biases login are missing;
- stay within the current W&B storage budget and report impending quota pressure;
- avoid destructive operations, uncontrolled resource exhaustion, and actions
  that could damage the machine or destroy unrecoverable work;
- ask before introducing a new paid external resource, publishing private data,
  or taking another action that requires authority beyond this repository; and
- never treat access to tooling, authorship, or a passing self-review as merge
  authority.

Agents may create branches, commit, push, and open pull requests. Human review
and merge is the default. A human may explicitly authorize an agent to merge one
named pull request or a bounded ticket or goal series. The authorization must
identify that scope and be recorded in both the pull request and model-run
record; it cannot be inferred from tool access or a general desire for autonomy.
The most recent human instruction controls, and ambiguity or revocation restores
the human-merge default.

Even with explicit authorization, an agent may self-merge only when every gate
below is satisfied for the exact head commit:

- an independent review returned `PASS` or a justified `PASS WITH NOTE` against
  the ticket, this philosophy, and applicable `CHECK.md` sections;
- every actionable finding has been repaired and independently re-reviewed; no
  blocking review decision, `CHANGES_REQUESTED` review, newer human objection,
  or review thread remains outstanding; and the agent has not dismissed a human
  review to clear the gate;
- branch-protection required contexts and applicable configured workflows and
  checks have been inventoried, every expected check is present and successful
  for the reviewed head, and no expected check is absent, pending, skipped,
  cancelled, or otherwise non-successful; a no-check state is accepted only with
  evidence that no check is required, configured, or expected, never merely
  because the current status list is empty;
- the pull request is up to date with its target branch, conflict-free, and
  reported mergeable;
- the model-run record, ledger, pull-request execution trail, validation
  evidence, risks, and authorization evidence are complete and agree; and
- the merging agent performs and records a final audit of these gates in the PR
  at the exact reviewed head, without creating a new unreviewed commit; and
- immediately before invoking merge, the merging agent re-fetches authorization,
  head and base identities, review decisions and newer objections, review
  threads, the expected-check inventory and exact-head statuses, and
  mergeability; compares them with the final audit; records that observation
  without changing the head; and aborts for the appropriate update,
  revalidation, or independent re-review if any compared field drifted.

An agent never uses administrator privileges, disables a rule, bypasses branch
protection, or force-merges to satisfy these gates. Self-merge is prohibited for
a change that contains or authorizes secrets or security-control changes,
publication of private data, a new paid resource, a destructive or unrecoverable
action, an unresolved legal or licensing question, or another externally
consequential protected action such as deployment, release, account/permission
change, or non-routine external action outside ordinary repository collaboration.
Routine PR creation, review comments, evidence updates, and issue/PR coordination
are ordinary repository collaboration, not prohibited external communication.
Protected actions require a human merge decision even if a broader series
authorization exists.

The pull request remains the handoff: it should be possible to review the
hypothesis, diff, run status, plots, result, integrity checks, and proposed next
step in a short session. This matters because routine human attention may be
limited to one or two hours per week. A policy-changing pull request that first
introduces agent self-merge cannot bootstrap its own authority; the preceding
human-only policy governs until a human merges that policy change.

## Simple does not mean improvised

The codebase should look like professional ML engineering while remaining small
enough to understand. We value:

- direct, readable implementations of the important model and training logic;
- clear boundaries between data, tokenization, model, optimization, evaluation,
  checkpointing, and experiment tracking;
- one reproducible package and dependency environment;
- Hydra as the single runtime and training configuration system;
- executable commands for canonical workflows;
- tests for mathematical, data, checkpoint, and training-loop invariants;
  and
- explicit errors and recovery paths for long-running work.

We avoid speculative abstractions, compatibility shims, duplicated execution
paths, and framework-building that does not serve a current experiment.
Configuration should expose scientifically meaningful choices without turning
every implementation detail into an option.

Notebooks are not source code or canonical workflows. A notebook may later be a
thin client for inspecting a trained model or plotting existing results, but
training, evaluation, and inference behavior must live in importable modules and
reproducible command-line entrypoints.

## Optimize from measurements

Performance engineering is part of the research. Native C or CUDA code, custom
kernels, precision changes, compilation, and data-pipeline work are welcome when
they address a measured bottleneck on the target machine.

Every optimization must preserve a clear reference implementation or reference
result, include correctness checks with appropriate tolerances, and report both
the speedup and its measurement conditions. Throughput that silently changes the
training objective or numerical behavior is not an optimization.

We profile before specializing, and we keep low-level performance code behind a
small interface so an experiment does not make the whole codebase harder to
change.

## What progress means

Progress is a sequence of trustworthy answers, not merely a lower loss curve.
The project advances when we can explain:

- what changed;
- why the experiment could test it;
- whether model quality, stability, or efficiency improved;
- what the result rules out;
- whether the evidence is reproducible and free of leakage; and
- what should happen next.

The final model matters. So does leaving behind a codebase and experimental
history from which another person—or another agent—can understand how it was
built and continue improving it.
