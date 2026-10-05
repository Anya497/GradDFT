# AGENTS.md

This file is a short TOC of the agent workflow. The project itself is
described in `README.md`, which is the entrypoint for navigating the
repository.

## Start here

1. Load the `workflow-management` skill
   (`.opencode/skills/workflow-management/SKILL.md`) first, before doing
   anything else.
2. Read `README.md` for project navigation, then `examples/` for worked usage.

## Main Principles

* This file is a short entry point for fast cold errors-free start.
* Only one source of truth. No duplicates. Each thing (in both code and documentation) described exactly once. Use generalization (especially for code), cross-references, links, other similar techniques to avoid duplicates and reuse staff.
* Source-of-truth hierarchy: code, scripts, CI configs > docs. If a fact can
  be extracted from code or scripts (e.g., CI configs), it is not duplicated
  in docs. Docs hold only what cannot be unambiguously reconstructed from
  code: design decisions, non-trivial constraints. Skills stay thin pointers
  to docs, code, or CI — they never re-describe them.
* Tools, not instructions. If you can do something with existing tool --- do it. No thinking, no long instructions, no manual analysis. You want to analyze code coverage? Just run coverage tool and analyze report. No workaround for regular tasks. If there is a tool for regular task it must be installed and configured appropriately.
* Never stay silent about problems. Any problem discovered or detected must be addressed. If it is in current scope or the fix is easy (e.g. a misprint), fix it — even if pre-existing. If it is significantly out of scope, report it to the user and propose creating an issue.
* Always learn, never forget — encode patterns before session ends.

## Repository Workflow Configuration

The integration branch is the long-lived personal branch `agent_settings`
(stacked on `main`); resolve it with the command in the `workflow-management`
skill, which is the single source of truth. Feature branches are cut from it,
one per task.

## Skills

### Domain

| Skill | When to use |
|---|---|
| `.opencode/skills/run-tests` | Running the test suite (`pytest`, per `.github/workflows/install_and_test.yml`) |
| `.opencode/skills/quality-gates` | The pre-merge gate that must pass |

### Workflow / process

| Skill | When to use |
|---|---|
| `.opencode/skills/workflow-management` | Driving the overall task loop |
| `.opencode/skills/planning` | Global plans, atomic subtasks, task authoring |
| `.opencode/skills/subtask-loop` | Executing one atomic subtask |
| `.opencode/skills/user-guidance-transfer` | Recording user guidance verbatim |
| `.opencode/skills/code-review` | Whole-repo review before merge |
