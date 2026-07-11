## Project preferences
- Use Hydra for runtime/training configuration; do not add a separate `config.py`.
- Prefer direct imports.
- Do not break tasks into subtasks unnecessarily.
- Unless explicitly requested, do not preserve backward compatibility; prefer the direct change over compatibility aliases or shims.

## Required implementation and review workflow

- Read the relevant `ROADMAP.md` ticket before changing code. When an implementation decision is ambiguous, use `PHILOSOPHY.md` as the decision policy instead of inventing a new project direction.
- Start and continuously update a draft pull request while implementing. The PR is the live handoff, not something written only after the work is finished. If PR creation is unavailable, prepare the complete PR body from `.github/pull_request_template.md` and state the blocker instead of inventing a PR URL.
- By default, use Luna or the available lightweight implementation model at `Extra High` reasoning or higher for the first implementation pass. After implementation, use an available heavier model at `Extra Thinking` for an independent review against `PHILOSOPHY.md`, the ticket acceptance criteria, and only the applicable sections of `CHECK.md`.
- If the required heavier model or reasoning mode is unavailable in the current runtime, record the review as `blocked` and prepare the full review handoff. Do not silently substitute the implementation pass for the independent heavy review or declare a passing verdict without it.
- Model names and reasoning modes vary by runtime. Record the exact model identifier and reasoning mode actually displayed by the runtime. Never infer or fabricate them; write `not exposed by runtime` when the value is unavailable.
- The post-implementation `CHECK.md` review is mandatory. It checks the real ML system, performance, DGX behavior, research integrity, and unnecessary complexity. It does not by itself require adding unit tests or generic software tests. Tests explicitly required by the roadmap ticket or needed to prove a mathematical/data invariant still remain in scope.
- If the heavy review returns `FAIL`, do not mark the ticket complete. Record what failed and the evidence, select the next implementation model deliberately, and hand it the failed review, relevant repository context, constraints, and a concrete repair request. Repeat implementation and independent review until the result is `PASS` or an explicitly justified `PASS WITH NOTE`.
- Track every implementation, review, and repair cycle in `docs/model-runs/<ticket>-<slug>.md` using `docs/model-runs/TEMPLATE.md`. Update the summary tables in `docs/model-runs/README.md` in the same PR.
- Every PR must include a link to its model-run record and a model execution trail showing phase, exact model, reasoning mode, outcome, and important findings. A PR with missing model provenance is incomplete.
- Record observable engineering rationale, changes, failures, evidence, and handoff context. Do not claim access to hidden chain-of-thought.
- A human reviews and merges the PR. Agents must not treat a passing model review as merge authority.
