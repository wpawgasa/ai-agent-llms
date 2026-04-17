---
name: code-implementation
description: Implement code changes with a structured workflow. Use when Claude needs to implement features, fix bugs, or make code changes in a repository. Includes repository exploration, branch creation, implementation planning, unit testing, coverage checks, documentation, and git commits. Ensures quality through testing and proper documentation.
---

# Code Implementation Skill

Structured workflow for implementing code changes with quality assurance.

## Workflow Overview

```
1. Understand → 2. Explore → 3. Branch → 4. Plan → 5. Implement → 6. Test → 7. Document → 8. Commit
```

---

## Step 1: Understand the Query

Before any code work, clarify the request:

**Classify the task type**:
| Type | Description | Branch Prefix |
|------|-------------|---------------|
| Feature | New functionality | `feature/` |
| Fix | Bug fix | `fix/` |
| Refactor | Code improvement, no behavior change | `refactor/` |
| Chore | Maintenance, config, dependencies | `chore/` |
| Docs | Documentation only | `docs/` |
| Test | Test additions/improvements | `test/` |
| Perf | Performance improvement | `perf/` |

**Identify**:
- What is the expected outcome?
- What files/modules are likely affected?
- Are there acceptance criteria?
- Any constraints (backward compatibility, performance)?

---

## Step 2: Explore Repository

### 2.1 Read CLAUDE.md First

```bash
cat CLAUDE.md
```

CLAUDE.md contains project-specific context: architecture, conventions, setup instructions, and guidelines. This is critical context.

### 2.2 Repository Structure

```bash
# Get directory overview
find . -type f -name "*.py" | head -50  # or relevant extension
ls -la
tree -L 2 -I 'node_modules|__pycache__|.git|venv|dist'
```

### 2.3 Understand Existing Patterns

```bash
# Check existing tests structure
ls -la tests/ test/ spec/ __tests__/

# Check for config files
ls -la *.json *.yaml *.toml pyproject.toml package.json

# Check for existing docs
ls -la docs/ doc/ README.md
```

### 2.4 Identify Related Code

```bash
# Search for related functions/classes
grep -r "function_name" --include="*.py" .
grep -r "ClassName" --include="*.ts" .
```

---

## Step 3: Create Branch

### 3.1 Ensure Clean State

```bash
git status
git stash  # if needed
```

### 3.2 Create Feature Branch

**Naming convention**: `{type}/{short-description}`

```bash
# Examples
git checkout -b feature/user-authentication
git checkout -b fix/null-pointer-exception
git checkout -b chore/upgrade-dependencies
git checkout -b refactor/extract-validation-logic
git checkout -b docs/api-documentation
git checkout -b test/add-integration-tests
git checkout -b perf/optimize-database-queries
```

**Branch name rules**:
- Lowercase only
- Use hyphens for spaces
- Keep concise but descriptive
- No special characters except hyphen

---

## Step 4: Plan Implementation

**Before writing code, create a plan**:

```markdown
## Implementation Plan

### Goal
[One sentence describing the outcome]

### Affected Files
- [ ] `path/to/file1.py` — [what changes]
- [ ] `path/to/file2.py` — [what changes]
- [ ] `tests/test_file1.py` — [new tests]

### Approach
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Edge Cases
- [Edge case 1]
- [Edge case 2]

### Testing Strategy
- Unit tests for: [components]
- Integration tests for: [workflows]
- Edge cases to cover: [list]
```

Share plan with user for confirmation before proceeding.

---

## Step 5: Implement

### 5.1 Follow Project Conventions

Match existing code style:
- Indentation, naming conventions
- Import organization
- Comment style
- Error handling patterns

### 5.2 Implement Incrementally

- Make small, focused changes
- Test each component as you go
- Keep changes related to the task

### 5.3 Code Quality Checklist

- [ ] Follows existing code patterns
- [ ] No hardcoded values (use constants/config)
- [ ] Proper error handling
- [ ] Type hints (if project uses them)
- [ ] Docstrings for public functions/classes
- [ ] No commented-out code
- [ ] No debug print statements

---

## Step 6: Unit Tests

### 6.1 Test File Location

Follow project convention, typically:
```
project/
├── src/module/feature.py
└── tests/module/test_feature.py
```

### 6.2 Test Structure

**Python (pytest)**:
```python
import pytest
from module.feature import FeatureClass

class TestFeatureClass:
    """Tests for FeatureClass."""
    
    def test_happy_path(self):
        """Test normal operation."""
        result = FeatureClass().method(valid_input)
        assert result == expected_output
    
    def test_edge_case_empty_input(self):
        """Test handling of empty input."""
        result = FeatureClass().method([])
        assert result == []
    
    def test_error_case_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid input"):
            FeatureClass().method(invalid_input)
```

**JavaScript/TypeScript (Jest)**:
```typescript
import { featureFunction } from './feature';

describe('featureFunction', () => {
  it('should handle normal input', () => {
    expect(featureFunction(validInput)).toBe(expectedOutput);
  });

  it('should throw on invalid input', () => {
    expect(() => featureFunction(invalidInput)).toThrow('Invalid');
  });
});
```

### 6.3 Test Coverage Requirements

**Run tests with coverage**:

```bash
# Python
pytest --cov=src --cov-report=term-missing
pytest --cov=src --cov-report=html

# JavaScript/TypeScript
npm test -- --coverage
npx jest --coverage
```

**Coverage targets**:
- New code: aim for 80%+ coverage
- Critical paths: 100% coverage
- Edge cases: explicitly tested

### 6.4 What to Test

| Test Type | What to Cover |
|-----------|---------------|
| Happy path | Normal input → expected output |
| Edge cases | Empty, null, boundary values |
| Error cases | Invalid input, exceptions |
| Integration | Component interactions |

---

## Step 7: Documentation

### 7.1 Create Summary Document

Create `docs/{feature-name}-summary.md`:

```markdown
# [Feature/Fix Name]

**Date**: YYYY-MM-DD  
**Branch**: `feature/feature-name`  
**Author**: Claude

## Overview

[1-2 paragraph description of what was implemented]

## Changes

### New Files
- `path/to/new_file.py` — [Purpose]

### Modified Files
- `path/to/modified.py` — [What changed]

### Deleted Files
- `path/to/removed.py` — [Why removed]

## Technical Details

[Implementation details, algorithms used, design decisions]

## Testing

- Unit tests: `tests/test_feature.py`
- Coverage: X%
- Test cases:
  - [Test case 1]
  - [Test case 2]

## Usage

```python
# Example usage
from module import feature
result = feature.do_something(input)
```

## Related

- Issue: #XXX (if applicable)
- Related docs: [links]
```

### 7.2 Update CLAUDE.md

Add relevant context for future development:

```markdown
## [Section for new feature/module]

- Location: `src/module/`
- Purpose: [what it does]
- Key patterns: [important conventions]
- Dependencies: [external deps]
- Testing: `pytest tests/module/`
```

**What to add to CLAUDE.md**:
- New architectural decisions
- Non-obvious implementation details
- Important conventions introduced
- Setup/configuration changes
- Known limitations or TODOs

---

## Step 8: Commit Changes

### 8.1 Review Changes

```bash
# Review all changes
git diff

# Review summary
git diff --stat
```

### 8.2 Stage Selectively

**Never use `git add -A` or `git add .`**

```bash
# Stage implementation files
git add src/module/feature.py
git add src/module/helper.py

# Stage tests
git add tests/module/test_feature.py

# Stage documentation
git add docs/feature-summary.md

# Stage CLAUDE.md update
git add CLAUDE.md

# Do NOT stage other markdown files
# git add CHANGELOG.md  # NO
```

### 8.3 Verify Staged Changes

```bash
git diff --cached
git diff --cached --stat
git status
```

### 8.4 Write Detailed Commit Message

```bash
git commit -m "<type>(<scope>): <subject>

<body>

<footer>"
```

**Example**:
```bash
git commit -m "feat(auth): implement JWT token refresh mechanism

Add automatic token refresh for expired JWT tokens:
- Create TokenRefreshService with configurable refresh window
- Add refresh token storage in secure HTTP-only cookies
- Implement silent refresh on 401 responses
- Add token refresh endpoint at POST /auth/refresh

Token refresh occurs 5 minutes before expiration to prevent
interrupted user sessions. Refresh tokens valid for 7 days.

Testing:
- Unit tests: 12 new tests, 100% coverage on new code
- Integration tests: 3 new tests for refresh flow

Docs: docs/jwt-refresh-summary.md

Closes #234"
```

---

## Quick Reference

| Step | Key Command |
|------|-------------|
| Read context | `cat CLAUDE.md` |
| Create branch | `git checkout -b feature/name` |
| Run tests | `pytest --cov=src` |
| Review changes | `git diff` |
| Stage file | `git add <file>` |
| Verify staged | `git diff --cached --stat` |
| Commit | `git commit -m "..."` |

## Quality Gates

Before committing, verify:

- [ ] All tests pass
- [ ] Coverage meets threshold (80%+)
- [ ] No linting errors
- [ ] Documentation created in `docs/`
- [ ] CLAUDE.md updated
- [ ] Only relevant files staged
- [ ] Commit message is detailed
