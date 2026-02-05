# How to Exit a Dart Application Unconditionally

## Overview

The `exit()` method terminates a running Dart VM by exiting the current program. It accepts a status code parameter where non-zero values typically indicate abnormal termination, similar to exit functionality in C/C++, Java.

## Key Concepts

### Syntax
```dart
exit(exit_code);
```

### Required Import
You must import the `'dart:io'` package to use this method.

### Exit Code Conventions

**Linux and OS X:**
- Exit codes must fall within the 0-255 range
- Values outside this range get masked to their lower 8 bits as unsigned values
- Example: -1 becomes 255

**Windows:**
- Accepts any 32-bit value
- Reserved codes exist for system errors
- Dart uses 254 for compile-time errors and 255 for runtime errors
- Recommendation: use codes 0-127 to avoid cross-platform compatibility issues

### Status Code Meanings
- `exit(0)` indicates successful termination
- Non-zero values generally signal unsuccessful termination

## Implementation Details

The method internally checks if exiting is allowed and uses `ArgumentError.checkNotNull()` to validate the code parameter before calling `_ProcessUtils._exit()`.

## Practical Example

```dart
import 'dart:io';

void main() {
  // This will be printed
  print("Hello GeeksForGeeks");

  // Exit with success code
  exit(0);

  // This will NOT be printed
  print("Good Bye GeeksForGeeks");
}
```

**Output:**
```
Hello GeeksForGeeks
```

## Important Notes

The `exit()` method terminates immediately without waiting for asynchronous operations to complete, making it useful for unconditional program termination.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/how-to-exit-a-dart-application-unconditionally/
- **Fetched**: 2026-01-27
