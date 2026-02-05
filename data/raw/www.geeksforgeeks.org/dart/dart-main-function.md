# Dart - main() Function Tutorial

## Overview

The `main()` function is a predefined method in Dart that serves as the mandatory entry point for any Dart application. Every Dart program requires this function to execute and is responsible for running all library and user-defined code.

## Syntax

```dart
void main() {
    // function body of main()
}
```

The function returns `void` and can include variable declarations, function declarations, and executable statements. Optional `List<String>` parameters allow external program control.

## Key Characteristics

- **Mandatory**: Required for program execution
- **Entry Point**: Where Dart applications begin running
- **Return Type**: Always `void`
- **Optional Parameters**: Accepts `List<String> arguments`

## Example 1: Basic Implementation

Create a file named `temp.dart`:

```dart
void main() {
    print("Main is the entry point!");
}
```

**Command to run:**
```
dart temp.dart
```

**Output:**
```
Main is the entry point!
```

## Example 2: Accepting Arguments

Create a file named `main.dart`:

```dart
main(List<String> arguments) {
    // printing the arguments along with length
    print(arguments.length);
    print(arguments);
}
```

**Command to run:**
```
dart main.dart Argument1 Argument2
```

**Output:**
```
2
[Argument1, Argument2]
```

## Practical Applications

The `main()` function enables developers to:
- Execute all library functions systematically
- Run user-defined statements and functions
- Control program behavior through command-line arguments
- Manage overall application flow and initialization

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-main-function/
- **Fetched**: 2026-01-27
