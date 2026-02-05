# Exception Handling in Dart

## Overview

An exception represents an error occurring during program execution. When an exception happens, the program's normal flow is disrupted, causing abnormal termination with error messages and stack traces displayed. Proper exception handling prevents application crashes.

## Built-in Exceptions

Dart provides several pre-defined exception classes, all inheriting from the `Exception` class. Common built-in exceptions include `FormatException`, `NoSuchMethodError`, and others specific to various operations.

## Try-On Block

The `try-on` block handles specific exception types:

```dart
try {
  // code that might throw an exception
} on SpecificException {
  // handles SpecificException only
}
```

**Example:**

```dart
void main() {
  String geek = "GeeksForGeeks";
  try {
    int result = int.parse(geek);
  } on FormatException {
    print('Error!! Can\'t parse non-integer input.');
  }
}
```

**Output:**
```
Error!! Can't parse non-integer input.
```

## Try-Catch Block

The `catch` block handles any exception type thrown in the try block:

```dart
try {
  // code that might throw an exception
} catch(e) {
  // handles any exception
  print(e);
}
```

**Example:**

```dart
void main() {
  String geek = "GeeksForGeeks";
  try {
    var geek2 = geek ~/ 0;
  } catch(e) {
    print(e);
  }
}
```

## Finally Block

The `finally` block executes regardless of whether an exception occurs, useful for cleanup operations:

```dart
try {
  // code that might throw an exception
} catch(e) {
  // exception handling
} finally {
  // always executes
}
```

**Example:**

```dart
void main() {
  int geek = 10;
  try {
    var geek2 = geek ~/ 0;
  } catch(e) {
    print(e);
  } finally {
    print("Execution complete");
  }
}
```

## Custom Exceptions

Create custom exceptions by implementing the `Exception` class:

```dart
class CustomException implements Exception {
  String toString() => 'Custom error message';
}
```

**Example:**

```dart
class Age implements Exception {
  String toString() => 'Your age is less than 18 :(';
}

void main() {
  int age1 = 20;
  int age2 = 10;

  try {
    check(age1);
    check(age2);
  } catch (e) {
    print(e);
  }
}

void check(int age) {
  if (age < 18) {
    throw new Age();
  } else {
    print("You are eligible!");
  }
}
```

## Key Points

- Use `try-on` for specific exception handling
- Use `try-catch` for general exception handling
- Use `finally` for guaranteed cleanup code
- Throw custom exceptions using the `throw` keyword
- Multiple exception handlers can be chained with multiple `on` or `catch` blocks

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/exception-handling-in-dart/
- **Fetched**: 2026-01-27
