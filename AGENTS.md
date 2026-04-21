# Repository Agent Notes

These notes apply to the entire repository.

## Compute Planning

- The user has approximately `$10,000` in Lambda Cloud credits available for this project.
- Do not assume heavy training must run on the user's laptop.
- For GPU-heavy work such as real training runs, multi-seed sweeps, pretraining, large ablations, or long evaluation jobs, plan for remote execution first.
- When remote compute is needed, pause and ask the user to start the Lambda Cloud machine and provide VM SSH access.
- Prefer using the laptop for local development, smoke tests, config validation, and small CPU-safe checks only.

## Research-First Planning

- Before implementing major model, pretraining, training, or evaluation work, check for recent external developments that may change the best plan.
- Explicitly look for maintained upstream GitHub repos, official implementations, recent papers, pretrained checkpoints, and actively used baselines relevant to the current story.
- Prefer adapting or integrating a strong upstream implementation when it is a better fit than rebuilding from scratch.
- Do not assume the best approach is the one already in local context; verify whether newer or better-supported options exist first.
- For architecture or pretrained-model decisions, include a short buy-vs-build check during planning, create-story, dev-story, and code-review work.

## Story Shipping Workflow

- After a story is done, package it on its own delivery branch, commit the changes, and open a PR before moving on.
- While that PR is under review, continue with the next planned story on a follow-up branch or continuation branch as directed by the user.
- When review comments come back, fix real issues, update the branch, and merge before treating the story as fully landed.
- External branch names must be product-focused and must not include internal workflow labels such as `bmad`, `epic`, or `story`.
- In external delivery artifacts such as commit messages, PR titles, PR bodies, PR comments, and reviewer replies, do not mention internal workflow terms such as BMAD, epic, or story.
- Keep external delivery language product-focused and implementation-focused only.
- After opening a PR, request automated review by commenting `@coderabbitai review` and `@greptileai`.
