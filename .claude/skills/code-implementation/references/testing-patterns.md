# Testing Patterns Reference

Language-specific testing patterns and coverage tools.

## Python

### pytest Setup

```bash
# Install
pip install pytest pytest-cov

# Run tests
pytest
pytest -v                          # verbose
pytest tests/test_specific.py      # specific file
pytest -k "test_name"              # by name pattern

# With coverage
pytest --cov=src --cov-report=term-missing
pytest --cov=src --cov-report=html  # generates htmlcov/
pytest --cov=src --cov-fail-under=80
```

### pytest Patterns

```python
import pytest
from unittest.mock import Mock, patch, MagicMock

# Fixtures
@pytest.fixture
def sample_data():
    return {"key": "value"}

@pytest.fixture
def mock_service():
    with patch("module.ExternalService") as mock:
        mock.return_value.fetch.return_value = {"data": "test"}
        yield mock

# Parameterized tests
@pytest.mark.parametrize("input,expected", [
    ("valid", True),
    ("invalid", False),
    ("", False),
])
def test_validation(input, expected):
    assert validate(input) == expected

# Exception testing
def test_raises_on_invalid():
    with pytest.raises(ValueError, match="must be positive"):
        process_value(-1)

# Async tests
@pytest.mark.asyncio
async def test_async_function():
    result = await async_fetch()
    assert result is not None
```

### Coverage Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src --cov-report=term-missing"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## JavaScript/TypeScript

### Jest Setup

```bash
# Install
npm install --save-dev jest @types/jest ts-jest

# Run tests
npm test
npx jest
npx jest --coverage
npx jest --watch
npx jest path/to/test.ts
```

### Jest Patterns

```typescript
import { functionUnderTest } from './module';

// Basic test
describe('functionUnderTest', () => {
  it('should return expected value', () => {
    expect(functionUnderTest('input')).toBe('output');
  });

  it('should handle edge cases', () => {
    expect(functionUnderTest('')).toBeNull();
    expect(functionUnderTest(null)).toBeUndefined();
  });
});

// Mocking
jest.mock('./dependency', () => ({
  fetchData: jest.fn().mockResolvedValue({ data: 'test' }),
}));

// Async tests
it('should fetch data', async () => {
  const result = await fetchData();
  expect(result).toEqual({ data: 'test' });
});

// Error testing
it('should throw on invalid input', () => {
  expect(() => processValue(-1)).toThrow('must be positive');
});

// Snapshot testing
it('should match snapshot', () => {
  expect(renderComponent()).toMatchSnapshot();
});
```

### Jest Configuration (jest.config.js)

```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: ['**/*.test.ts', '**/*.spec.ts'],
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts'],
  coverageThreshold: {
    global: { branches: 80, functions: 80, lines: 80, statements: 80 },
  },
};
```

---

## Go

### Testing Commands

```bash
# Run tests
go test ./...
go test -v ./...
go test -run TestSpecific ./...

# With coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
go tool cover -func=coverage.out
```

### Go Test Patterns

```go
package module_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestFunction(t *testing.T) {
    result := FunctionUnderTest("input")
    assert.Equal(t, "expected", result)
}

// Table-driven tests
func TestValidation(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected bool
    }{
        {"valid input", "valid", true},
        {"empty input", "", false},
        {"invalid input", "invalid", false},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Validate(tt.input)
            assert.Equal(t, tt.expected, result)
        })
    }
}

// Error testing
func TestErrorCase(t *testing.T) {
    _, err := ProcessValue(-1)
    require.Error(t, err)
    assert.Contains(t, err.Error(), "must be positive")
}
```

---

## Rust

### Testing Commands

```bash
# Run tests
cargo test
cargo test -- --nocapture  # show println output
cargo test test_name       # specific test

# With coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --out Html
```

### Rust Test Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        assert_eq!(function_under_test("input"), "expected");
    }

    #[test]
    fn test_edge_case() {
        assert!(validate("").is_err());
    }

    #[test]
    #[should_panic(expected = "must be positive")]
    fn test_panic() {
        process_value(-1);
    }

    // Async tests (requires tokio)
    #[tokio::test]
    async fn test_async() {
        let result = async_function().await;
        assert!(result.is_ok());
    }
}
```

---

## Coverage Targets

| Code Type | Target | Rationale |
|-----------|--------|-----------|
| New feature code | 80%+ | Ensure core logic tested |
| Bug fix | 100% on fix | Prevent regression |
| Critical paths | 100% | Payment, auth, data integrity |
| Utility functions | 90%+ | Highly reusable |
| UI components | 70%+ | Harder to unit test |
| Generated code | Skip | Auto-generated |

## Test Naming Conventions

```
test_<function>_<scenario>_<expected_result>

# Examples
test_validate_empty_input_returns_false
test_process_negative_value_raises_error
test_fetch_valid_id_returns_user
test_create_duplicate_name_fails
```
