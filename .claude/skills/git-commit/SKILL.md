---
name: git-commit
description: Commit code changes to git with careful review and detailed commit messages. Use when Claude needs to stage and commit code changes to a git repository. This skill ensures changes are reviewed with git diff before staging, excludes markdown files (except README.md), and generates comprehensive commit messages following conventional commit standards.
---

# Git Commit Skill

Commit code changes with careful review and detailed, meaningful commit messages.

## Workflow

### Step 1: Review Changes

Before staging, always review what has changed:

```bash
# View all unstaged changes
git diff

# View summary of changed files
git diff --stat

# View changes for a specific file
git diff <filepath>
```

Analyze the diff output to understand:
- What files were modified, added, or deleted
- The nature of each change (bug fix, feature, refactor, etc.)
- Whether changes are related or should be separate commits

### Step 2: Stage Files Selectively

Stage files individually or by pattern. Never use `git add -A` or `git add .`.

```bash
# Stage specific files
git add <filepath1> <filepath2>

# Stage by pattern (e.g., all Python files in a directory)
git add src/*.py
```

**Markdown exclusion rule:** Do not stage markdown files (*.md) except for `README.md`:

```bash
# Correct: stage README.md
git add README.md

# Incorrect: do not stage other markdown files
# git add CHANGELOG.md  # NO
# git add docs/*.md     # NO
```

### Step 3: Verify Staged Changes

Before committing, verify what will be committed:

```bash
# View staged changes
git diff --cached

# View staged files summary
git diff --cached --stat

# View status
git status
```

### Step 4: Write Detailed Commit Message

Commit with a comprehensive message following this structure:

```bash
git commit -m "<type>(<scope>): <subject>

<body>

<footer>"
```

#### Commit Message Format

**Line 1 - Header (required):**
- `<type>`: feat, fix, refactor, docs, style, test, chore, perf, ci, build
- `<scope>`: Component/module affected (optional but recommended)
- `<subject>`: Imperative, lowercase, no period, max 50 chars

**Body (required for non-trivial changes):**
- Explain WHAT changed and WHY (not how)
- Wrap at 72 characters
- Use bullet points for multiple changes
- Reference related issues/tickets

**Footer (when applicable):**
- Breaking changes: `BREAKING CHANGE: <description>`
- Issue references: `Fixes #123`, `Closes #456`

#### Example Commit Messages

**Simple fix:**
```bash
git commit -m "fix(auth): resolve token expiration check

The previous implementation compared timestamps incorrectly,
causing tokens to expire 1 hour early. Changed comparison
operator from < to <= to include the boundary condition.

Fixes #234"
```

**Feature with multiple changes:**
```bash
git commit -m "feat(api): add rate limiting to endpoints

Implement rate limiting across all public API endpoints:
- Add sliding window rate limiter middleware
- Configure limits per endpoint in config.yaml
- Return 429 status with Retry-After header when exceeded
- Add rate limit headers to all responses

Rate limits are configurable per-environment and default
to 100 requests/minute for authenticated users.

Closes #567"
```

**Refactoring:**
```bash
git commit -m "refactor(database): extract connection pooling logic

Move connection pool management from individual repositories
into dedicated ConnectionPool class. This centralizes:
- Pool size configuration
- Connection health checks  
- Retry logic with exponential backoff

No functional changes. Reduces code duplication across
5 repository classes by ~200 lines."
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `git diff` | Review all unstaged changes |
| `git diff --stat` | Summary of changed files |
| `git diff --cached` | Review staged changes |
| `git add <file>` | Stage specific file |
| `git status` | View current state |
| `git commit -m "..."` | Commit with message |

## Common Patterns

**Staging related changes together:**
```bash
git diff src/auth/
git add src/auth/login.py src/auth/token.py
git commit -m "fix(auth): improve token validation..."
```

**Checking before commit:**
```bash
git diff --cached --stat
# Verify only intended files are staged
# Verify no markdown files (except README.md) are included
git commit -m "..."
```

**Handling README.md updates:**
```bash
# README.md is the only markdown file that should be committed
git add README.md
git commit -m "docs(readme): update installation instructions..."
```
