# Documentation Templates

Templates for implementation summary documents and CLAUDE.md updates.

## Summary Document Template

Location: `docs/{feature-name}-summary.md`

```markdown
# [Feature/Fix Name]

**Date**: YYYY-MM-DD  
**Branch**: `{type}/{branch-name}`  
**Type**: Feature | Fix | Refactor | Chore

---

## Overview

[2-3 sentences describing what was implemented and why]

## Problem Statement

[What problem did this solve? What was broken/missing before?]

## Solution

[High-level description of the approach taken]

---

## Changes

### New Files

| File | Purpose |
|------|---------|
| `src/module/new_file.py` | [Description] |
| `tests/test_new_file.py` | [Test coverage for new_file] |

### Modified Files

| File | Changes |
|------|---------|
| `src/module/existing.py` | [What was changed and why] |
| `config/settings.yaml` | [Configuration changes] |

### Deleted Files

| File | Reason |
|------|--------|
| `src/deprecated.py` | [Why removed] |

---

## Technical Details

### Architecture

[Describe architectural decisions, patterns used]

### Key Components

**ComponentName**
- Purpose: [What it does]
- Location: `src/module/component.py`
- Dependencies: [What it relies on]

### Algorithms/Logic

[Explain any non-trivial algorithms or business logic]

### Configuration

[New configuration options, environment variables, etc.]

```yaml
# Example configuration
feature:
  enabled: true
  timeout: 30
```

---

## Testing

### Test Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| `src/module/feature.py` | 95% | Missing edge case X |
| `src/module/helper.py` | 100% | Fully covered |

### Test Cases

- ✅ Happy path: [description]
- ✅ Edge case - empty input: [description]
- ✅ Edge case - boundary values: [description]
- ✅ Error handling: [description]

### How to Run Tests

```bash
# Run all tests for this feature
pytest tests/module/test_feature.py -v

# Run with coverage
pytest tests/module/ --cov=src/module --cov-report=term-missing
```

---

## Usage

### Basic Example

```python
from module import Feature

# Initialize
feature = Feature(config)

# Use
result = feature.process(data)
```

### Advanced Example

```python
# With custom configuration
feature = Feature(
    timeout=60,
    retry_count=3,
    on_error=custom_handler
)

# Process with options
result = feature.process(
    data,
    validate=True,
    transform=True
)
```

---

## API Reference (if applicable)

### `ClassName.method_name(params)`

**Parameters:**
- `param1` (str): Description
- `param2` (int, optional): Description. Default: 10

**Returns:**
- `ReturnType`: Description

**Raises:**
- `ValueError`: When [condition]

---

## Migration Notes (if applicable)

[Any steps needed to migrate from old behavior]

```python
# Before
old_function(x, y)

# After
new_function(x, y, new_required_param=value)
```

---

## Known Limitations

- [Limitation 1 and potential workaround]
- [Limitation 2]

## Future Improvements

- [ ] [Potential enhancement 1]
- [ ] [Potential enhancement 2]

---

## Related

- Issue: #XXX
- PR: #YYY
- Related docs: [links]
- External references: [links]
```

---

## CLAUDE.md Update Templates

### New Feature Section

```markdown
## [Feature Name]

**Location**: `src/module/`

### Purpose
[What this feature does]

### Key Files
- `feature.py` — Main implementation
- `helpers.py` — Utility functions
- `types.py` — Type definitions

### Patterns
- [Pattern 1 used in this module]
- [Pattern 2]

### Dependencies
- External: `library_name>=1.0`
- Internal: `src/common/utils`

### Testing
```bash
pytest tests/module/ -v
```

### Usage
```python
from module import feature
result = feature.process(data)
```
```

### Bug Fix Note

```markdown
## Bug Fix: [Brief Description] (YYYY-MM-DD)

**Issue**: [What was broken]
**Cause**: [Root cause]
**Fix**: [How it was fixed]
**Location**: `src/module/file.py`

**Regression Prevention**: Test added in `tests/test_file.py::test_specific_case`
```

### Architecture Decision

```markdown
## Architecture Decision: [Topic]

**Date**: YYYY-MM-DD
**Status**: Accepted

### Context
[Why this decision was needed]

### Decision
[What was decided]

### Consequences
- [Positive consequence]
- [Trade-off or negative consequence]
```

### Configuration Documentation

```markdown
## Configuration: [Feature Name]

**File**: `config/feature.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable feature |
| `timeout` | int | `30` | Timeout in seconds |

### Environment Variables
- `FEATURE_ENABLED` — Override enabled setting
- `FEATURE_TIMEOUT` — Override timeout
```

---

## Commit Message Templates

### Feature

```
feat(scope): add [feature description]

Implement [detailed description]:
- [Change 1]
- [Change 2]
- [Change 3]

Testing:
- Unit tests: X new tests
- Coverage: Y% on new code

Docs: docs/feature-summary.md
Closes #XXX
```

### Bug Fix

```
fix(scope): resolve [bug description]

Root cause: [explanation]

Fix: [what was changed]
- [Specific change 1]
- [Specific change 2]

Testing:
- Added regression test in test_xxx.py
- Verified fix with [steps]

Fixes #XXX
```

### Refactor

```
refactor(scope): [refactor description]

Motivation: [why refactor was needed]

Changes:
- [Change 1]
- [Change 2]

No functional changes. All existing tests pass.
```
