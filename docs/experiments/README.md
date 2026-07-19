# Experiment Records

Every experiment uses one roadmap ticket, one focused branch, and one explicit
hypothesis. Name branches `codex/<ticket-lower>-<slug>`, for example
`codex/exp-001-review-record`. Give the experiment record the matching
`<TICKET>-<slug>.md` basename. If the question changes, stop and open a new
ticket/branch/record rather than growing the original experiment in place.

## Current research state

- **Authoritative baseline:** `none — wave-4 readiness gates remain unmet`.
- **Evidence:** `origin/main` at `ed53daa3df5a790ae9c81d4cb44b93d654eb8532`
  records 17 Done tickets, with `VAL-001` and `WB-001` Ready. A run must not be
  called the real pretraining baseline until the remaining wave-4 validation,
  tracking, benchmark, DGX measurement, and operations gates are complete.
- **Next critical-path question:** `VAL-001` — can fixed held-out Japanese and
  English corpora produce trustworthy, reproducible validation evidence?

The baseline statement above is deliberately stronger than "the code runs."
Change it only when merged evidence satisfies the roadmap readiness gates and
identifies the baseline commit and run.

## Experiment evidence and review evidence

`docs/experiments/<ticket>-<slug>.md` is the scientific and operational record.
It declares the hypothesis and budget before a run, then preserves every
attempt's resolved configuration, identities, measurements, failures, evidence,
comparison, and conclusion.

The pull request is the implementation and review handoff. It preserves the
change rationale, `/review` findings, repair cycles, validation, risks, and
final verdict. It does not replace experiment evidence; the PR and experiment
record cross-link when a change contains a consequential run.

## Fresh-agent read path

A fresh agent should read, in order:

1. `PHILOSOPHY.md` for the research decision policy.
2. `ROADMAP.md` for the current baseline decision, dependencies, ticket scope,
   and next Ready question.
3. This file for the record contract and current baseline pointer.
4. The selected file in `docs/experiments/` for run evidence.
5. The draft PR for the live diff, review findings, validation, unresolved
   uncertainty, and the human decision requested.

Copy `TEMPLATE.md` before doing consequential work. Fill predeclared fields
before running, append an attempt for every launch (including negative or
aborted launches), and never delete an attempt.
