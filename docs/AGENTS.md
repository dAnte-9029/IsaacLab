
Project-Documentation Guidelines

These instructions apply to project state, architecture, design decisions, implementation plans, audits, handoffs, and research-facing documentation.

1. Documentation roles

Use the following document types:

PROJECT_STATE.md: short current-state index.
architecture/: descriptions of the current implemented system.
decisions/ADR-*.md: approved design decisions and their rationale.
plans/: proposed or active implementation plans.
audits/: evidence-based code audits.
handoffs/: stage-end context for the next task or conversation.

Do not use AGENTS.md as a substitute for these documents.

2. Source-of-truth rules
   Code and tests describe implemented behavior.
   ADRs describe approved design decisions.
   PROJECT_STATE.md identifies the currently active baseline and next stage.
   Handoffs describe what was completed and what remains.

When documentation conflicts with code, identify the conflict rather than silently editing history.

3. ADR discipline

An ADR must contain:

context;
decision;
alternatives considered;
consequences;
assumptions;
validation requirements;
conditions that would trigger reconsideration.

Do not silently rewrite an accepted ADR to represent a new decision. Create a superseding ADR or clearly mark the old ADR as superseded.

4. Project-state discipline

Keep PROJECT_STATE.md concise.

It should contain:

stable branch and commit or tag;
completed stage;
active stage;
required reading;
known blockers;
exact next task.

Do not turn it into a chronological diary.

5. Audit requirements

Code-audit statements must cite concrete repository evidence:

file path;
class or function;
relevant configuration;
test or command when available.

Separate:

observed facts;
interpretations;
unresolved questions;
recommendations.

Do not claim a simulation, test, or experiment was run unless it was actually executed.

6. Handoff requirements

A stage handoff should record:

branch and commit;
objective;
approved assumptions;
files changed;
tests run;
results;
known limitations;
unresolved human decisions;
exact next task;
allowed and prohibited modification scope;
reproduction commands.
7. Research-claim boundaries

Documentation must distinguish among:

force-level validation;
numerical integration;
task-level simulation demonstration;
controller evaluation;
model validation;
sim-to-real evidence;
real-flight validation.

Do not upgrade an integration demonstration into a model-validation claim
