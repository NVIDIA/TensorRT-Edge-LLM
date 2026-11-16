<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Install the Edge-LLM agent skills

Public skills live in `skills/edge-llm-*/SKILL.md`. `.claude/skills` and
`.agents/skills` are symlinks to `skills/`, so a host started inside a clone
can discover them without any install.

## Choose the level

| Situation | Level |
|---|---|
| Working inside a clone | Repo-local: nothing to install |
| Working outside any clone | User-level: symlink each `skills/edge-llm-*` directory into the host's user skills directory |

A user-level link and the repo-local copy collide whenever the user works inside
a clone. Which copy a host loads is host behavior that this repository does not
define; check the host documentation, or ask the agent in a fresh session which
`SKILL.md` path it loaded.

## Repo-local

1. Confirm that `.claude/skills` and `.agents/skills` are symlinks whose
   target is `../skills`.
2. Start a fresh host session from the clone root and ask it to report the
   loaded path for `edge-llm-install`. It must resolve to this clone. Repeat for
   any skill needed by the task.
3. If a name is missing, the host does not discover that directory or does not
   follow the symlink; check the host's documentation for its project skills
   location. Change the symlinks or the host configuration only with the user's approval.

## User-level

Ask before writing to the user skills directory. Use the location the host's
documentation gives for user-level skills (Claude Code: `~/.claude/skills`) and
report it. Set `DEST` to that directory and `REPO` to the clone's absolute path
(a relative path produces dangling links).

Take the skill names from the `skills/edge-llm-*` directories of the clone
(`REPO`). For each name, confirm `DEST/<name>` does not exist, then create one
link with the user's approval:

```bash
ln -s "$REPO/skills/<name>" "$DEST/<name>"
```

Skip any name that already exists and report it as not replaced. Record each
link you create; removal touches only those. Existing entries are never
replaced. Start a fresh session and list skills to verify. If the host
does not list a symlinked skill, report it; copying the directory instead
loses update tracking and is the user's decision.

## Version provenance

User-level links follow whatever the clone has checked out. Report:

```bash
git -C "$REPO" rev-parse HEAD
git -C "$REPO" describe --tags --always --dirty
```

## Update

1. Require a clean clone (`git -C "$REPO" status --porcelain` is empty) and the
   user's consent, then `git -C "$REPO" pull --ff-only`.
2. Rerun the install loop to link skills added upstream.
3. List dangling links that point into the clone, and remove them only with
   consent:

   ```bash
   find "$DEST" -maxdepth 1 -xtype l -lname "$REPO/skills/*"
   ```

4. Report the provenance commands above and start a fresh session to reload.

## Remove

Remove only links recorded at install time, one at a time, and never a
directory or a link you did not create. For each recorded path, confirm that
`readlink "$DEST/<name>"` points into `"$REPO"/skills/`, then ask the user to
approve deleting that single link and use the host's file-deletion operation.
A path that is not a link, or whose target is elsewhere, is reported and left
alone.

Repo-local discovery needs no removal; it ends when the host runs outside the
clone.

## Evidence

Report the level, host, `DEST`, each linked or removed path, provenance output,
and the in-session skill listing. A created symlink is not evidence that the
host loads it.
