# Experiment Records

Every experiment uses one roadmap ticket, one focused branch, and one explicit
hypothesis. Name branches `codex/<ticket-lower>-<slug>`, for example
`codex/exp-001-review-record`. Give both the experiment and model-run records
the matching `<TICKET>-<slug>.md` basename. If the question changes, stop and open a new
ticket/branch/record rather than growing the original experiment in place.

## Current research state

- **Authoritative baseline:** `none — readiness gates unmet`.
- **Evidence:** `origin/main` at `a05eb1de5656643757a1c3d98047c98dedea8bfa`
  contains the `ROADMAP.md` AS-IS snapshot and readiness decision. It says not
  to call a run a baseline until the listed CUDA, data, tokenizer, split,
  training-loop, checkpoint, generation, logging, and overfit gates exist.
- **Next unanswered question:** `DATA-001` — can packed streaming train every
  intended next-token transition exactly once, including window boundaries?

The baseline statement above is deliberately stronger than "the code runs."
Change it only when merged evidence satisfies the roadmap readiness gates and
identifies the baseline commit and run.

## Two records with different purposes

`docs/experiments/<ticket>-<slug>.md` is the scientific and operational record.
It declares the hypothesis and budget before a run, then preserves every
attempt's resolved configuration, identities, measurements, failures, evidence,
comparison, and conclusion.

`docs/model-runs/<ticket>-<slug>.md` is model-execution provenance for the code
and documentation change. It records which implementation and independent
review models were invoked, their exposed reasoning modes, outcomes, repair
cycles, and review verdict. It does not replace experiment evidence; the two
records cross-link when a PR contains a consequential run.

## Fresh-agent read path

A fresh agent should read, in order:

1. `PHILOSOPHY.md` for the research decision policy.
2. `ROADMAP.md` for the current baseline decision, dependencies, ticket scope,
   and next Ready question.
3. This file for the record contract and current baseline pointer.
4. The selected file in `docs/experiments/` for run evidence.
5. The linked file in `docs/model-runs/` for implementation/review provenance.
6. The draft PR for the live diff, validation, unresolved uncertainty, and the
   human decision requested.

Copy `TEMPLATE.md` before doing consequential work. Fill predeclared fields
before running, append an attempt for every launch (including negative or
aborted launches), and never delete an attempt.
