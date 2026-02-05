# Delete File in Dart - Comprehensive Guide

## Overview

This tutorial covers file deletion in Dart using the `File` class and its methods from the `dart:io` library.

## Basic File Deletion

### Simple Deletion Method

To delete a file, use the `File` class with the `deleteSync()` method:

```dart
import 'dart:io';

void main() {
  File file = File('test.txt');
  file.deleteSync();
  print('File deleted.');
}
```

**Output:**
```
File deleted.
```

### Important Consideration

The tutorial notes: "If you try to delete a file that does not exist, then it will throw an exception." This means attempting to remove a non-existent file will cause your program to error.

## Safe Deletion with Existence Check

To prevent exceptions, verify the file exists before attempting deletion:

```dart
import 'dart:io';

void main() {
  File file = File('test.txt');

  if (file.existsSync()) {
    file.deleteSync();
    print('File deleted.');
  } else {
    print('File does not exist.');
  }
}
```

**Output (when file doesn't exist):**
```
File does not exist.
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `deleteSync()` | Removes the file synchronously |
| `existsSync()` | Checks if file exists before deletion |

This approach ensures robust file handling without unexpected runtime errors.

---

## Source

- **URL**: https://dart-tutorial.com/file-handling-in-dart/delete-file-in-dart/
- **Fetched**: 2026-01-27
