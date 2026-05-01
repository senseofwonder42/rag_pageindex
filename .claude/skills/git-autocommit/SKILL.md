---
name: git-autocommit
description: >
  Commits and pushes changes to Git. Run only when the user explicitly asks to commit or push,
  or invokes /git-autocommit. Do NOT run proactively or automatically after tasks.
---

# git-autocommit

Skill that automatically commits and pushes changes to Git at the end of each task.

## When this skill runs

Run this skill at the end of **any task that modifies files** in a Git repository:
- After implementing a feature or fixing a bug
- After refactoring or reorganizing code
- After updating documentation or configuration
- After any multi-file change session

## Workflow

Follow these steps in order, without skipping any.

### Step 1 — Check Git status

Run the following to understand what changed:

```bash
git status
git diff --stat
```

If `git status` returns "nothing to commit, working tree clean", tell the user there's nothing to commit and stop here.

If the directory is not a Git repository, tell the user and stop.

### Step 2 — Analyze the changes

Read the diff to understand what was actually changed:

```bash
git diff --cached
git diff
```

Group the changes by type to prepare the commit message:
- New files → likely `feat:` or `chore:`
- Modified logic files → likely `feat:`, `fix:`, or `refactor:`
- Test files → `test:`
- Documentation files → `docs:`
- Config files (pyproject.toml, .env, etc.) → `chore:`
- Deleted files → `chore:` or `refactor:`

### Step 3 — Propose a commit message

Generate a Conventional Commits message based on the analysis.

**Format:**
```
<type>(<scope>): <short description in imperative mood>

<optional body explaining WHY, not WHAT — only if the change is non-obvious>
```

**Types to use:**
| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `chore` | Build, config, dependency updates |
| `style` | Formatting, linting (no logic change) |
| `perf` | Performance improvement |

**Scope:** Use the module or folder name affected (e.g., `feat(tree): ...`, `refactor(api): ...`). Omit if changes span many areas.

**Short description:** Imperative mood, max 72 chars, no period at the end.

**Example output to show the user:**

```
📋 Résumé des changements :
  - src/pageindex/tree.py      (modifié)
  - src/pageindex/search.py    (modifié)
  - tests/test_tree.py         (nouveau)

💬 Message de commit proposé :
  refactor(tree): replace dict-based nodes with typed dataclasses

  Improves type safety and IDE autocompletion. TreeNode now uses
  @dataclass with explicit field types.

🌿 Branche : refactor/architecture
📤 Destination : origin/refactor/architecture
```

Then ask explicitly:
> "✅ Je committe et pousse avec ce message ? (oui / modifie le message)"

Wait for the user's answer before proceeding.

### Step 4 — Apply the user's decision

**If the user confirms (yes, oui, ok, go, etc.):**

```bash
git add -A
git commit -m "<the proposed message>"
git push
```

**If the user wants to edit the message:**
- Accept the corrected message
- Then run the same commands with the new message

**If the user says no / cancel:**
- Acknowledge and stop. Do not commit anything.

### Step 5 — Report the result

After a successful commit + push, confirm with a short summary:

```
✅ Commit effectué et poussé !
   Commit : abc1234
   Branche : refactor/architecture → origin/refactor/architecture
```

If the push fails (e.g., upstream conflict), report the error clearly and suggest:
```bash
git pull --rebase origin <branch>
# then retry the push
```

## Important rules

- **Never commit without showing the user what will be committed first.** The confirmation step is mandatory.
- **Never force-push** (`git push --force`) unless the user explicitly asks for it.
- **If there are untracked files** that look important (e.g., new .py files), mention them explicitly so the user can decide to include them.
- **Use the branch that's currently checked out.** Never switch branches automatically.
