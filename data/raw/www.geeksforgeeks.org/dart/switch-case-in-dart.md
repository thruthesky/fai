# Switch Case in Dart - Complete Tutorial

## Overview

Switch-case statements are a great tool for managing multiple conditions in a clean and organized way and function as superior alternatives to lengthy if-else chains for improving code readability.

## Syntax Structure

```dart
switch (expression) {
  case value1:
    // Body of value1
    break;

  case value2:
    // Body of value2
    break;

  default:
    // Body of default case
    break;
}
```

The default case executes when no other case conditions match, though it remains optional.

## Key Rules for Switch Statements

- Multiple cases are permitted, but values must remain unique
- Case statements require only constants—variables and expressions are invalid
- Flow control via `break` statements is mandatory within cases
- The default case is optional
- Nested switch statements are supported

## Normal Switch-Case Example

```dart
void main() {
  int gfg = 1;
  switch (gfg) {
    case 1: {
      print("GeeksforGeeks number 1");
    }
    break;

    case 2: {
      print("GeeksforGeeks number 2");
    }
    break;

    case 3: {
      print("GeeksforGeeks number 3");
    }
    break;

    default: {
      print("This is default case");
    }
    break;
  }
}
```

**Output:**
```
GeeksforGeeks number 1
```

## Nested Switch-Case Example

```dart
void main() {
  int gfg1 = 1;
  String gfg2 = "Geek";

  switch (gfg1) {
    case 1: {
      switch (gfg2) {
        case 'Geek': {
          print("Welcome to GeeksforGeeks");
        }
      }
    }
    break;

    case 2: {
      print("GeeksforGeeks number 2");
    }
    break;

    default: {
      print("This is default case");
    }
    break;
  }
}
```

**Output:**
```
Welcome to GeeksforGeeks
```

## Summary

Switch statements provide an efficient mechanism for handling multiple constant conditions. Unlike optional break statements that may cause unexpected fall-through behavior, Dart requires explicit flow control. Default cases offer a safety net for unmatched conditions, and nested implementations work effectively despite infrequent usage.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/switch-case-in-dart/
- **Fetched**: 2026-01-27
