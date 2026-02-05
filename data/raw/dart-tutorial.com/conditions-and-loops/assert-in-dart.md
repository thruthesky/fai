# Assert in Dart

## Overview

"While coding, it is hard to find errors in big projects, so dart provide a **assert** method to check for the error." The assert function validates conditions during development and raises errors when conditions evaluate to false.

## Syntax

Assert statements can be used with or without custom error messages:

```dart
assert(condition);
// or
assert(condition, "Your custom message");
```

## Example 1: Basic Assert Usage

This demonstrates assert functionality without a custom message:

```dart
void main() {
   var age = 22;
   assert(age!=22);
}
```

**Output:**
```
Uncaught Error: Assertion failed
```

## Example 2: Assert with Custom Message

This shows how to include a descriptive error message:

```dart
void main() {
   var age = 22;
   assert(age!=22, "Age must be 22");
}
```

**Output:**
```
Uncaught Error: Assertion failed: "Age must be 22"
```

## Running Files in Assert Mode

To enable assertion checking when executing Dart files locally, use:

```bash
dart --enable-asserts file_name.dart
```

## Important Note

"The **assert(condition)** method only runs in development mode. It will throw an exception only when the condition is false." Production environments typically ignore assertions for performance reasons.

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/assert-in-dart/
- **Fetched**: 2026-01-27
