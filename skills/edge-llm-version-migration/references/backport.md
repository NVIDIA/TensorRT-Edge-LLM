<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backporting a fix to a release branch

Read this when a fix must land on a `release/<version>` branch.

## Target branch

- `CONTRIBUTING.md` and `AGENTS.md` define the target: changes go to `main`
  unless they fix a bug reported on a release branch, in which case they go to
  that release branch.
- Do not assume how fixes reach `main`. Check for release-branch merges into
  it (`git log --merges --oneline origin/main --grep "^Merge branch 'release/"`)
  before filing a separate forward-port.
- Branch from the tip of the remote release branch after `git fetch`, never
  from a stale local copy or from `main`.

## Preflight, per commit, before touching the worktree

1. **Already applied?** `git cherry -v origin/release/<version> <source-tip> <source-base>`
   marks patch-equivalent commits with `-`. Patch IDs change when a fix was
   squashed or adapted, so also search the release branch by subject and by
   the changed hunk (`git log origin/release/<version> --grep`, `git log -S`).
2. **Applies cleanly?**
   `git merge-tree --write-tree --name-only --merge-base=<commit>^ origin/release/<version> <commit>`
   (git 2.40+) exits 0 when clean; on 1, the conflicting paths follow the
   tree ID. Nothing is written to the worktree. On older git, try
   `git cherry-pick --no-commit` in a scratch worktree instead.
3. **Depends on absent code?** For each conflict, find whether the fix relies
   on a feature, file, flag, or plugin that does not exist on the release
   branch (`git show origin/release/<version>:<path>`). Do not backport that
   feature to make the fix apply; adapt the fix minimally or report it as not
   backportable.

## Apply

- `git cherry-pick -x -s <commit>` (DCO sign-off per `AGENTS.md`) on the
  non-merge commits in order (`git rev-list --no-merges --reverse <base>..<tip>`);
  a single merge commit needs `-m 1`. Resolve conflicts against release-branch code, not
  by copying `main`'s version of the file.
- Leave version stamps (`tensorrt_edgellm/_version.py`, `cpp/common/version.h`,
  `project(VERSION)` in `CMakeLists.txt`) and `CHANGELOG.md` alone unless asked.
- A clean `git cherry-pick` does not run pre-commit hooks, so run them afterwards
  and fold any fixes into the same commit:

  ```bash
  pre-commit run --files $(git diff-tree --no-commit-id --name-only -r HEAD)
  git add -u && git commit --amend --no-edit   # keeps the -x trailer and sign-off
  ```

## Per-commit outcome table

| Commit | Subject | Outcome | Notes |
|---|---|---|---|
| `<sha>` | ... | clean / adapted / already applied / not backportable | conflict paths, adaptation, missing dependency |

## Verify on the release branch

Build the release branch from a clean build directory, then run the
narrowest test that fails without the fix plus export -> build -> inference
for an affected model (`edge-llm-e2e-test`, `edge-llm-convert-model`). A clean
cherry-pick is not evidence; the release branch may lack code the fix assumed.

Report the release branch tip used, the outcome table, the commands run, and
the test and inference evidence.
