# Python Cheat Sheet

---

# 🐍 Python Cheat Sheet: Data Types, Data Structures, OOP & Core Concepts

## 1. **Import Convention**

Most core data structures and OOP features in Python are built-in. However:

```python
# Common imports
from enum import Enum  # For Enum classes
import collections  # For namedtuple, defaultdict, deque

```

---

## 2. **Core Functions/Classes/Concepts Table**

| **Concept/Function/Method** | **Example Usage** | **Short Description** |
| --- | --- | --- |
| `list`, `tuple`, `set`, `dict` | `my_list = [1,2,3]` | Built-in data structures. |
| `namedtuple` | `Point = namedtuple('Point', 'x y')` | Immutable, lightweight class alternative. |
| `defaultdict` | `d = defaultdict(int)` | Dict with default value factory. |
| `deque` | `dq = deque([1,2,3])` | Double-ended queue for fast appends/pops. |
| `@dataclass` | `@dataclass class Person:` | Auto-generates `__init__`, `__repr__`. |
| `Enum` | `class Color(Enum):` | Defines constant enums. |
| `lambda` | `add = lambda x, y: x+y` | Anonymous inline function. |
| `map`, `filter`, `reduce` | `map(str, [1,2,3])` | Functional programming tools. |
| `*args, **kwargs` | `def func(*args, **kwargs):` | Flexible argument passing. |
| `__str__`, `__repr__` | `def __str__(self):` | Controls how objects print. |
| `is`, `==` | `a is b` vs `a == b` | Identity vs equality check. |
| `isinstance`, `issubclass` | `isinstance(obj, Class)` | Type/Inheritance checking. |
| OOP Inheritance | `class Child(Parent):` | Extends behavior of parent class. |
| Polymorphism | `def func(obj): obj.action()` | Same method name, diff. behavior. |
| Multiple Inheritance | `class C(A, B):` | Inherits from multiple parents. |
| Decorators | `@decorator` | Wraps a function to extend behavior. |
| Generators | `yield x` | Memory-efficient iterators. |
| Context Managers | `with open() as f:` | Handles setup/cleanup automatically. |
| `property` | `@property` | Getter/setter without explicit method calls. |
| Comprehensions | `[x for x in range()]` | One-liners for creating lists, sets, dicts. |
| `type()` | `type(obj)` | Returns object's type. |
| `dir()` | `dir(obj)` | Lists attributes/methods of object. |

---

## 3. **Common Operations & Their Usage**

| **Operation** | **Example** | **Purpose** |
| --- | --- | --- |
| Convert list to set | `set([1,2,2,3])` | Remove duplicates. |
| Sorting dict by values | `sorted(d.items(), key=lambda x: x[1])` | Custom sorting. |
| List flattening | `[i for sublist in l for i in sublist]` | Flatten nested lists. |
| Merge dicts | `{**dict1, **dict2}` | Combine two dictionaries. |
| Reverse dict | `{v:k for k,v in d.items()}` | Invert key-values. |
| Count items | `collections.Counter(list)` | Frequency count. |
| Memory-efficient loop | `for x in generator:` | Avoid large memory usage. |
| Safe dict access | `d.get('key', default)` | Prevent KeyError. |
| Enumerate list | `for i, v in enumerate(lst):` | Index+value in loop. |
| Zipping iterables | `zip(list1, list2)` | Pair elements. |
| Swap variables | `a, b = b, a` | Clean swapping. |

---

## 4. **Useful Tips/Pro Tips/Best Practices**

- **Use `__repr__` for debugging, `__str__` for readable print.**
- **Prefer `is` only when checking for `None`, singletons.**
- **Use `@dataclass` over writing repetitive class boilerplate.**
- **Use list comprehensions over manual loops for concise, faster code.**
- **Generators (`yield`) shine when dealing with large datasets.**
- **`defaultdict` avoids boilerplate when populating dicts with lists/counters.**
- **Avoid mutable default arguments (`def func(x=[])`) — use `None` and check inside.**
- **Enumerate is more Pythonic than manual index tracking.**
- **Use `with` statements for all file and resource management.**
- **Break classes logically — don’t overdo inheritance; prefer composition.**

---

## 5. **Integration Example: ML & Data Science**

```python
import pandas as pd
import numpy as np
from enum import Enum

# Enum + Pandas example
class Gender(Enum):
    MALE = 0
    FEMALE = 1

df = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Gender': [Gender.FEMALE, Gender.MALE]
})

# Apply function with lambda
df['Gender_str'] = df['Gender'].apply(lambda x: x.name)

```

---

## 6. **Mini Project / Small Exercises**

1. **Class Exercise:**
    
    Write a `BankAccount` class with attributes `owner`, `balance`, `__str__`, deposit & withdraw methods. Add inheritance to create a `SavingsAccount` with an interest method.
    
2. **Polymorphism Practice:**
    
    Create a `Shape` class and two subclasses `Circle` and `Square`. Each subclass should implement an `area()` method differently. Write a function to take any shape and print area (demonstrate polymorphism).
    
3. **Lambda/Comprehension Drill:**
    
    Given a list of numbers, remove duplicates, sort in descending order, and square each number — all in one line using comprehensions/lambda.
    
4. **Enum Usage:**
    
    Define an `Enum` class for days of the week. Write a function that takes a string input and returns the corresponding Enum or raises an error if invalid.
    
5. **Data Structures Challenge:**
    
    Given a string, count the frequency of each character using `defaultdict` and `Counter`, compare which is more efficient.
    
6. **Decorator Task:**
    
    Write a decorator that logs the execution time of any function.
    
7. **Context Manager Creation:**
    
    Write your own custom context manager using `__enter__` and `__exit__` to time code blocks.
    
8. **Generator Task:**
    
    Create a generator that yields Fibonacci numbers up to a given `n`.
    
9. **Multiple Inheritance Scenario:**
    
    Model a `FlyingCar` class that inherits from `Car` and `Aircraft` classes, each having distinct methods. Handle method conflicts carefully.
    
10. **Property Practice:**
    
    Create a class `Temperature` that stores Celsius but allows getting/setting Fahrenheit using `@property`.
    

---

---

# 🐍 **ULTIMATE PYTHON CHEAT SHEET (Intermediate to Advanced)**

## 1. **Import Convention (Minimal)**

```python
python
CopyEdit
from collections import defaultdict, deque, Counter, namedtuple
from dataclasses import dataclass
from enum import Enum
import heapq

```

## 2. **Core Data Structures & Methods**

### 🔹 **List (Mutable, Ordered)**

```python
python
CopyEdit
lst = [1, 2, 3]
lst.append(4)          # Add element at end (O(1))
lst.pop()              # Remove last element (O(1))
lst.insert(1, 10)      # Insert at index (O(n))
lst.remove(2)          # Remove first occurrence of value (O(n))
lst.sort()             # Sort in-place (O(n log n))
lst.reverse()          # Reverse in-place
lst.count(2)           # Count occurrences
lst.index(3)           # Find index of value
sum(lst), min(lst), max(lst)  # Aggregates
lst[::2]               # Slicing
[x*x for x in lst]     # List comprehension

```

**🔥 LeetCode Must-Know:**

- `.sort()` for sorting arrays
- `.reverse()`
- Two-pointer problems often use slicing or index tricks.

---

### 🔹 **Tuple (Immutable)**

```python
python
CopyEdit
tup = (1, 2, 3)
tup.count(1)
tup.index(2)

```

---

### 🔹 **Set (Unique Elements, Unordered)**

```python
python
CopyEdit
s = {1, 2, 3}
s.add(4)
s.remove(2)
s.discard(10)  # Won't error if missing
s.pop()        # Removes arbitrary element
s.union({5,6})
s.intersection({2,3,4})
s.difference({1,4})
s.issubset({1,2,3,4})

```

**🔥 LeetCode Usage:**

- Quick membership check → `if val in set` (O(1))

---

### 🔹 **Dictionary (Key-Value Store)**

```python
python
CopyEdit
d = {'a': 1, 'b': 2}
d['c'] = 3
d.get('d', 0)               # Safe key access
d.pop('b')                  # Remove key
d.setdefault('e', 0)        # Sets default if key missing
d.update({'f': 5})          # Merge dict
for k, v in d.items():      # Loop over key-value pairs
    print(k, v)
{v: k for k, v in d.items()}  # Reverse dict

```

**🔥 LeetCode Usage:**

- Use hash maps (dict) for:
    - Two Sum
    - Counting frequency (like Counter)
    - Grouping (e.g., group anagrams)

---

### 🔹 **Deque (Double-Ended Queue)**

```python
python
CopyEdit
dq = deque([1,2,3])
dq.append(4)
dq.appendleft(0)
dq.pop()
dq.popleft()
dq.rotate(1)

```

**🔥 Usage:**

- BFS traversal
- Sliding window problems (fixed size windows)

---

### 🔹 **defaultdict & Counter**

```python
python
CopyEdit
d = defaultdict(int)  # Auto-default value
d['x'] += 1

c = Counter('abca')
c.most_common(2)      # Top n elements
list(c.elements())    # Recreate iterable

```

**🔥 LeetCode Usage:**

- Frequency counting
- Quick anagram checks
- Sliding window character counts

---

## 3. **Python OOP & Inheritance Master Cheat Sheet**

## 1. **OOP Core Components Recap**

**Class Syntax:**

```python
class ClassName:
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2

    def method_name(self):
        return self.attr1

```

---

## 2. **Inheritance**

### 🔥 **Basic Inheritance:**

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"

dog = Dog("Fido")
print(dog.speak())  # Fido barks

```

- **Child class inherits all attributes & methods from parent.**
- Child can **override** methods.

---

### 🔥 **Use `super()` Properly:**

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Cat(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Calls parent constructor
        self.breed = breed

```

- Always prefer `super()` to avoid hardcoding parent class names (especially useful in **multiple inheritance**).

---

### 🔥 **Multiple Inheritance:**

```python
class Flyer:
    def fly(self):
        return "Flying"

class Swimmer:
    def swim(self):
        return "Swimming"

class Duck(Flyer, Swimmer):
    pass

d = Duck()
print(d.fly(), d.swim())

```

- **Method Resolution Order (MRO):**
Python checks parent classes **left to right**.
Check MRO:
    
    ```python
    print(Duck.__mro__)
    
    ```
    

---

### 🔥 **`isinstance()` & `issubclass()`**

```python
isinstance(d, Duck)          # True
isinstance(d, Flyer)         # True
issubclass(Duck, Flyer)      # True

```

- Check if object/class inherits from another.

---

### 🔥 **Abstract Base Class (ABC) & Interfaces**

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def move(self):
        pass

class Car(Vehicle):
    def move(self):
        return "Drive"

c = Car()

```

- Prevents instantiating incomplete base classes.
- Forces child classes to implement methods.

---

### 🔥 **Polymorphism:**

```python
class Shape:
    def area(self):
        pass

class Circle(Shape):
    def area(self):
        return "Circle area"

class Square(Shape):
    def area(self):
        return "Square area"

def print_area(shape: Shape):
    print(shape.area())

print_area(Circle())  # Circle area
print_area(Square())  # Square area

```

- Same interface (`area()`), **different behavior per subclass.**

---

### 🔥 **`__str__` vs `__repr__`:**

```python
class Person:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Person: {self.name}"

    def __repr__(self):
        return f"Person('{self.name}')"

p = Person("John")
print(str(p))    # Person: John
print(repr(p))   # Person('John')

```

- **`__str__`** → Readable, user-friendly.
- **`__repr__`** → Unambiguous, dev/debug-friendly (ideally, valid Python code).

---

### 🔥 **Properties (`@property` Decorator)**

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

```

- Control attribute access like a method but use it as `.fahrenheit`.

---

## 3. **Best Practices (Advanced OOP)**

- Use **inheritance only when there is an "is-a" relationship.**
    
    Prefer **composition** over inheritance when behavior doesn't match hierarchy.
    
- Keep your base classes **simple, abstract, and reusable**.
- Avoid **deep inheritance chains** (prefer max 2-3 levels).
- Use `super()` instead of explicitly referencing parent class names.
- Always implement `__repr__` for better debugging.

---

## 4. **Common Interview/LeetCode Patterns Using OOP & Inheritance**

### **1. Factory Pattern (Using Polymorphism)**

```python
class Shape:
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        return "Drawing Circle"

class Square(Shape):
    def draw(self):
        return "Drawing Square"

def shape_factory(shape_type):
    shapes = {'circle': Circle, 'square': Square}
    return shapes[shape_type]()

```

---

### **2. Strategy Pattern Example**

```python
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid {amount} by Credit Card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid {amount} by PayPal"

```

---

## 5. **Exercises**

1. **Create a base class `Employee` with attributes name & salary.**
    
    Derive `Manager` & `Developer` classes with overridden `work()` methods.
    
2. **Implement a class hierarchy with an abstract class `Vehicle`.**
    
    Add `Car` and `Bike` subclasses, each implementing a `move()` method.
    
3. **Design a `Logger` class using `__repr__` and `__str__` so that debug logs are unambiguous but user output is clear.**
4. **Demonstrate multiple inheritance with `Bird` inheriting from `Flyer` and `Animal`. Add an MRO check.**
5. **Implement a class with `@property` that converts temperature between Celsius and Fahrenheit.**

---

**Want me to package this focused OOP cheat sheet into a ready-to-download PDF or structured markdown for reference?**

---

## 4. **Functional Programming - lambda**

```python
python
CopyEdit
# Lambda
add = lambda x, y: x + y

# Map / Filter / Reduce
map_obj = map(str, [1, 2, 3])             # ['1', '2', '3']
filter_obj = filter(lambda x: x % 2, [1,2,3])  # Odd nums
from functools import reduce
reduce(lambda x,y: x+y, [1,2,3,4])        # Sum = 10

# Enumerate
for idx, val in enumerate(['a', 'b']):
    print(idx, val)

# Zip
for a, b in zip([1,2,3], ['a','b','c']):
    print(a, b)

```

---

## 5. **Common LeetCode/Real-World Operations**

✅ **Reverse a List:**

```python
python
CopyEdit
lst[::-1] or lst.reverse()

```

✅ **Check Anagram:**

```python
python
CopyEdit
Counter(s1) == Counter(s2)

```

✅ **Sliding Window:**

```python
python
CopyEdit
for i in range(len(arr)-k+1):
    window = arr[i:i+k]

```

✅ **Two Pointer Approach:**

```python
python
CopyEdit
left, right = 0, len(arr) - 1

```

✅ **Prefix Sum:**

```python
python
CopyEdit
prefix = [0]*len(arr)
prefix[0] = arr[0]
for i in range(1, len(arr)):
    prefix[i] = prefix[i-1] + arr[i]

```

✅ **Matrix Transpose:**

```python
python
CopyEdit
zip(*matrix)

```

✅ **Sort Dict by Value:**

```python
python
CopyEdit
sorted(d.items(), key=lambda x: x[1], reverse=True)

```

✅ **Heap Usage:**

```python
python
CopyEdit
heapq.heappush(h, val)
heapq.heappop(h)

```

---

## 6. **Useful Tips & Best Practices**

- Use `enumerate()` instead of manual index tracking.
- Avoid mutable default args (`def func(x=[])` → BAD).
- Always prefer list comprehensions for simple loops.
- Use `Counter`, `defaultdict`, `set()`—LeetCode gold.
- For resource management, always use `with` statement.
- Keep `__repr__` developer-focused, `__str__` user-friendly.

---

## 7. **Mini Exercises**

1. Implement `BankAccount` class with `deposit`, `withdraw`, and print methods.
2. Write a function to remove duplicates **in-place** from a sorted list.
3. Use a dict to solve Two Sum problem.
4. Write a decorator that logs function execution time.
5. Create a custom context manager to time code execution.
6. Write a generator for Fibonacci numbers.
7. Group strings by length using `defaultdict`.
8. Use deque to implement BFS on a graph.
9. Sort elements by frequency using Counter.
10. Implement OOP with polymorphism: base class `Shape`, subclasses `Circle`, `Square`.

---

## 3. **Common Operations & Usage: LeetCode + Real-World Focus**

| **Operation** | **Example** | **Notes** |
| --- | --- | --- |
| Reverse a list | `lst[::-1]`, `lst.reverse()` | Slicing or in-place. |
| Check if two strings anagrams | `Counter(s1) == Counter(s2)` | Fast O(n) check. |
| Sliding window | `for i in range(len(s) - k + 1): window = s[i:i+k]` | Classic substring/array window. |
| Two pointer | `left, right = 0, len(arr)-1` | Used in sorted arrays. |
| Hash set for uniqueness | `set(s)` | Quick lookup O(1). |
| Heap operations (via `heapq`) | `import heapq; heapq.heappush(h, val)` | Min-heaps, useful in k-largest problems. |
| Merge sorted arrays | `heapq.merge(arr1, arr2)` | Efficient merging. |
| Prefix sum | `prefix[i] = prefix[i-1] + nums[i]` | O(1) range sum queries. |
| Dict for frequency | `d = defaultdict(int)` | Classic count + lookup. |
| Stack via list | `stack.append()` + `stack.pop()` | LIFO behavior. |
| Queue via deque | `deque.append()` + `popleft()` | BFS, task scheduling. |
| Sort dict by value/key | `sorted(d.items(), key=lambda x: x[1])` | Frequently used. |
| Remove duplicates sorted list | Two pointers, skip duplicates. | O(n) time, O(1) space. |
| Matrix transpose | `zip(*matrix)` | Elegant row/column swap. |

## 4. **Useful Tips / Best Practices**

- **Use `enumerate()` in loops for index-value pairs.**
- **Default mutable arguments = 🚩. Always use `None`!**
- **Use `sorted()` instead of `.sort()` if you don't want to modify original list.**
- **Use list comprehensions for speed and readability.**
- **Use `collections.Counter` instead of manually counting.**
- **Leverage `set()` for O(1) lookups when checking membership.**
- **Use `with` statement everywhere you handle files or resources.**
- **Use `__repr__` for debugging-friendly printouts.**
- **Familiarize with `heapq`, `deque`, `defaultdict` — LeetCode gold.**

---

## 🐍 Python Exceptions Cheat Sheet

Absolutely! Here's a **beginner-to-intermediate friendly, comprehensive, and highly practical cheat sheet on Python Exceptions**:

## 🚀 Import Convention

```python
# No special imports required, exceptions are part of core Python.

```

---

## 📋 Core Functions/Classes/Concepts Table

| Concept Name | Example Usage | Short Description |
| --- | --- | --- |
| `try` / `except` | `try: x = 1/0 except ZeroDivisionError: print("Cannot divide by 0")` | Wrap code that might cause an exception. |
| `except ExceptionType` | `except ValueError:` | Handles a specific type of exception. |
| `except` (generic) | `except:` | Catches **all** exceptions. **Use with caution!** |
| `else` | `try: ... else: print("No Exception!")` | Runs if **no exception** occurred in `try` block. |
| `finally` | `finally: print("Cleanup!")` | Always runs, regardless of exceptions. Good for cleanup. |
| `raise` | `raise ValueError("Invalid value!")` | Manually trigger exceptions. |
| `assert` | `assert x > 0, "x must be positive"` | Quick sanity checks; raises `AssertionError` if condition fails. |
| Custom Exceptions (subclassing) | `class MyError(Exception): pass` | Define your own exception types. |

---

## 🔥 Common Operations & Their Usage

### 1. **Catching Specific Errors**

```python
try:
    num = int(input("Enter a number: "))
except ValueError:
    print("That's not a valid number!")

```

**Why:** Helps avoid program crashes due to incorrect user input.

---

### 2. **Catching Multiple Exception Types**

```python
try:
    file = open('data.txt')
    result = 10 / 0
except (FileNotFoundError, ZeroDivisionError) as e:
    print(f"Error: {e}")

```

**Why:** Useful when dealing with multiple possible issues.

---

### 3. **Using `else` and `finally` Together**

```python
try:
    print("Trying something...")
except Exception:
    print("An error occurred.")
else:
    print("Everything went fine!")
finally:
    print("Always runs (e.g., closing files).")

```

---

### 4. **Raising Your Own Errors**

```python
def withdraw(balance, amount):
    if amount > balance:
        raise ValueError("Insufficient funds")
    return balance - amount

```

---

### 5. **Custom Exception Example**

```python
class InvalidAgeError(Exception):
    pass

def check_age(age):
    if age < 0:
        raise InvalidAgeError("Age can't be negative!")

```

---

## 💡 Useful Tips / Pro Tips / Best Practices

- **Always catch specific exceptions**, NOT the generic `except:` unless absolutely necessary.
- **Use `finally` for cleanup tasks**: closing files, releasing resources, etc.
- **Don’t abuse `assert` in production code** — disableable with `python -O` (optimized mode).
- **Document your custom exceptions well**; helps debugging.
- Use **exception chaining** to preserve the original error:
    
    ```python
    try:
        do_something()
    except Exception as e:
        raise NewError("Something failed") from e
    
    ```
    

---

## 🛠️ (Optional) Integration Example

**Pandas Example:**

```python
import pandas as pd

try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("File not found. Please check the path.")

```

**Why:** Exception handling when reading/writing files is crucial when working with data.

---

### 🎯 **Scenario: Reading a file and performing division based on user input**

```python
python
CopyEdit
try:
    # Try to open a file
    with open('numbers.txt', 'r') as f:
        data = f.read()

    # Try to convert user input to integer
    num = int(input("Enter a number to divide by: "))

    # Try to perform division
    result = 100 / num
    print(f"Result is: {result}")

except FileNotFoundError:
    print("Error: The file 'numbers.txt' was not found.")

except ValueError:
    print("Error: Invalid input! Please enter an integer.")

except ZeroDivisionError:
    print("Error: Cannot divide by zero!")

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    print("Operation attempted, whether it succeeded or not.")

```

---

### **Why multiple exceptions are needed here:**

| Exception Type | When it Happens |
| --- | --- |
| `FileNotFoundError` | File `numbers.txt` is missing |
| `ValueError` | User inputs something that's **not an integer** |
| `ZeroDivisionError` | User inputs `0`, causing division by zero |
| Generic `Exception` | Catches **any unexpected error** (like permission errors, etc.) |

---

### ✅ **Why not just one broad `except`?**

- **Specific exceptions give better error messages.**
- Prevents catching unrelated errors accidentally (bad debugging practice).
- Easier to debug and maintain.

---

## 📝 Mini Project / Small Exercises

1. **User Input Validator:**
    - Write a function that asks a user to input an integer.
    - Catch exceptions if they enter non-integers.
2. **File Reader:**
    - Open a file provided by the user.
    - Handle cases where the file doesn't exist.
3. **Safe Division Function:**
    - Function that takes two numbers.
    - Catches division by zero and returns `None` instead of crashing.
4. **Custom Exception:**
    - Create a `PasswordTooShortError`.
    - Raise it if a password length is <8.
5. **List Index Checker:**
    - Ask user for an index.
    - Handle `IndexError` if they choose an invalid index.
6. **Multipurpose Exception Logger:**
    - Wrap some risky operations (file reading, type conversion).
    - Log exception types and messages to a `.log` file.
7. **Context Manager Cleanup:**
    - Use `try`/`except`/`finally` to ensure a file is always closed, even after an exception.

---
