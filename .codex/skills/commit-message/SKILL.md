---
name: commit-message
description: Draft repository-style git commit messages. Use when Codex needs to write, refine, review, or validate a commit message for code changes, especially when the repository expects structured multi-section messages with explicit motivation, concrete change summaries, and validation details.
---

# Commit Message

Draft commit messages. Keep the message concrete, explain why the change exists, and record validation when it materially helps reviewers.

## Workflow

1. Start from the template in `references/commit-template.md`.
2. Review the staged or requested changes with `git status --short` and a focused diff.
3. Draft a message that only includes sections that add information.

## Message Style

- Use the structure from `references/commit-template.md`.
- Use a short subject line in the form `<type>(<scope>): <summary>`.
- Prefer a blank line after the subject, then section headers such as `Why:`, `What:`, `Validation:`, and `Notes:`.
- Write section bodies as flat bullet lists with concrete statements.
- Keep wording factual and specific to the actual change. Avoid filler such as "misc fixes" or "update code."

## Section Rules

- Start from the template in `references/commit-template.md`.
- Keep `Why:` when motivation or constraints matter.
- Keep `What:` when there are concrete implementation or behavior changes to summarize.
- Keep `Validation:` when commands were run, tests were skipped for a meaningful reason, or validation details help reviewers.
- Keep `Notes:` only for caveats, follow-ups, environment limitations, or intentionally skipped work.
- Omit any section that would otherwise be empty, redundant, or filled with placeholders. Do not write `N/A`.

## Validation Guidance

- Prefer exact commands in backticks when validation was performed.
- If an important test could not run, either record that in `Validation:` or `Notes:` with the concrete reason.
- Do not invent validation. If nothing was run and the section would add no value, omit it.

## Output

- Return the final commit message as plain text ready to pass to `git commit`.
- When asked to improve an existing message, preserve correct facts and only tighten structure or wording.
