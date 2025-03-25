# Python pytest Cheat Sheet

---

## 📌 Topic Overview

### 🔹 Brief Description:

`pytest` is a powerful and flexible testing framework for Python. It allows you to write simple unit tests as well as complex functional tests in a clean, scalable way. It's widely used in Python development due to its readability, fixtures, and plugin system.

### 🔹 Key Concepts:

| Concept | Description |
| --- | --- |
| Test Function | A function prefixed with `test_` that gets run by `pytest`. |
| Assertion | Uses Python’s `assert` to validate expected vs. actual behavior. |
| Fixture | A function to set up and tear down state for tests. |
| Parametrize | Decorator to run a test with multiple sets of data. |
| Test Discovery | `pytest` automatically finds tests following naming conventions. |

### 🔹 Real-World Applications:

- Unit testing functions/modules
- Integration testing APIs or pipelines
- Regression testing for ML/DL workflows
- TDD (Test-Driven Development)

---

## ⚙️ Import Convention / Setup

### 🔸 Installation:

```bash
pip install pytest

```

### 🔸 Project Structure:

```
project/
│
├── test_example.py       ← test file (must start or end with "test")
├── example.py            ← your source code

```

### 🔸 Running Tests:

```bash
pytest                      # run all tests in current directory
pytest test_example.py      # run specific test file
pytest -k "function_name"   # run specific test by name match
pytest -v                   # verbose mode

```

---

## 🧰 Core Functions / Classes / Concepts

| Name | Example Usage | Short Description | Input/Output | Notes |
| --- | --- | --- | --- | --- |
| `assert` | `assert x == 2` | Basic assertion for expected behavior | Input: condition; Output: pass/fail | Use Python's built-in `assert` |
| `@pytest.mark.parametrize` | `@pytest.mark.parametrize("a,b", [(1,2), (3,4)])` | Test same function with multiple inputs | Input: arg names + list of values | Boost test coverage |
| `@pytest.fixture` | `@pytest.fixture\ndef db(): return DB()` | Setup/teardown reusable resources | Input: none; Output: resource | Can be scoped per function, module, etc. |
| `pytest.raises()` | `with pytest.raises(ValueError): func()` | Ensure code raises expected exception | Input: Exception; Output: context manager | Great for testing invalid inputs |
| `conftest.py` | Define shared fixtures across files | Auto-loaded helper file for fixtures | — | Place in test root directory |

---

## 🔄 Common Operations & Their Usage

### 🔸 1. Basic Unit Test

```python
def add(x, y):
    return x + y

def test_add():
    assert add(1, 2) == 3

```

✅ When to use: For isolated, deterministic functions.

---

### 🔸 2. Parameterized Test

```python
import pytest

@pytest.mark.parametrize("x,y,result", [(1, 2, 3), (4, 5, 9)])
def test_add_param(x, y, result):
    assert add(x, y) == result

```

✅ When to use: To avoid repeating the same test logic with different inputs.

---

### 🔸 3. Using Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    return {"user": "alice", "id": 1}

def test_sample_user(sample_data):
    assert sample_data["user"] == "alice"

```

✅ When to use: For setting up test dependencies.

---

### 🔸 4. Testing Exceptions

```python
def divide(a, b):
    return a / b

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)

```

✅ When to use: To ensure your code handles errors gracefully.

---

### 🔸 5. Skipping Tests

```python
import pytest

@pytest.mark.skip(reason="Pending bug fix")
def test_not_ready():
    assert False

```

✅ When to use: To temporarily disable failing tests.

---

## 💡 Useful Tips, Pro Tips & Best Practices

### 🔹 Best Practices:

- Use descriptive test names.
- One assert per test is ideal (not mandatory).
- Keep fixtures modular and reusable.
- Run tests frequently during development.

### 🔹 Troubleshooting Tips:

- Use `s` to see `print()` statements: `pytest -s`
- Use `x` to stop after first failure: `pytest -x`
- Use `-maxfail=3` to stop after 3 failures.

### 🔹 Performance Optimization:

- Use `pytest-xdist` for parallel execution: `pip install pytest-xdist`, then `pytest -n auto`.

---

## 🔗 Integration & Interoperability

### 🔸 PyTorch Example:

```python
import torch
import pytest

def test_tensor_addition():
    t1 = torch.tensor([1.0])
    t2 = torch.tensor([2.0])
    assert (t1 + t2).item() == 3.0

```

### 🔸 Pandas Example:

```python
import pandas as pd

def test_column_exists():
    df = pd.DataFrame({"name": ["Alice", "Bob"]})
    assert "name" in df.columns

```

---

## 🧪 Mini Project / Small Exercises

| Task | Description | Hint |
| --- | --- | --- |
| 1 | Write tests for a basic calculator (add, subtract, multiply, divide) | Test edge cases like divide by 0 |
| 2 | Write parameterized tests for string reverse function | Include empty and unicode strings |
| 3 | Create a fixture that sets up a sample DB or dict | Use a dictionary as mock DB |
| 4 | Test a function that raises an exception for invalid input | Use `pytest.raises()` |
| 5 | Test reading from a file (mock file read) | Use `tmp_path` fixture |
| 6 | Skip a test based on condition (e.g., OS) | Use `@pytest.mark.skipif(...)` |
| 7 | Write tests that run in parallel | Use `pytest-xdist` |
| 8 | Write integration tests for a Flask API route | Use test client from Flask |
| 9 | Test a class method with a fixture-based setup | Create fixture to return class instance |
| 10 | Test performance using time constraints | Use `pytest-benchmark` plugin |

---

## 🧠 Advanced Concepts & Extensions (Optional)

- **Plugins**: Explore `pytest-cov`, `pytest-mock`, `pytest-django`
- **Hooks & Customization**: Modify behavior using `conftest.py` hooks
- **Code Coverage**: `pip install pytest-cov`, then `pytest --cov=your_module`

---

## 📚 Additional Resources & References

- 📘 Official Docs: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)
- 🧪 Plugin Index: [https://docs.pytest.org/en/stable/plugins.html](https://docs.pytest.org/en/stable/plugins.html)
- 📺 RealPython Tutorial: [https://realpython.com/pytest-python-testing/](https://realpython.com/pytest-python-testing/)

---

## ❓ FAQs / Common Pitfalls

| Question | Answer |
| --- | --- |
| Why are my tests not running? | Ensure test file and function names start with `test_`. |
| Can I use `unittest` with `pytest`? | Yes, `pytest` is compatible with `unittest`. |
| How do I test classes? | Define test methods inside test classes prefixed with `Test`. |

### 🚫 Pitfalls to Avoid:

- Using `print()` instead of asserts.
- Forgetting to prefix test functions with `test_`.
- Mixing too much logic into fixtures.

---

## 🔁 Summary & Quick Recap

### ✅ Key Takeaways:

- `pytest` is concise, powerful, and beginner-friendly.
- Use fixtures for reusable setup logic.
- Use parameterization to test multiple inputs easily.
- Integration with other libraries (e.g., pandas, torch) is seamless.

---

### 🧾 Quick Reference:

```bash
# Install
pip install pytest

# Run all tests
pytest

# Verbose
pytest -v

# Run specific test
pytest test_file.py::test_func

# Parametrize
@pytest.mark.parametrize("x,y", [(1,2), (3,4)])

# Fixtures
@pytest.fixture
def sample(): return data

# Expect Exception
with pytest.raises(Exception):
    call()

```

---