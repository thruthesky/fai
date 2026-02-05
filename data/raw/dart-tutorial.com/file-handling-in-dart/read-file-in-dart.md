# Reading Files in Dart: Comprehensive Guide

## Introduction to File Handling

"File handling is an important part of any programming language." File operations enable developers to interact with the filesystem, a crucial capability for most applications.

## Basic File Reading

### Reading Complete File Content

To read an entire file, use the `File` class from the `dart:io` library with the `readAsStringSync()` method.

**Example: Reading test.txt**

```dart
import 'dart:io';

void main() {
  File file = File('test.txt');
  String contents = file.readAsStringSync();
  print(contents);
}
```

This code reads a complete file and prints its contents to the console.

## Retrieving File Information

Access metadata about files using dedicated methods:

```dart
import 'dart:io';

void main() {
  File file = File('test.txt');

  print('File path: ${file.path}');
  print('File absolute path: ${file.absolute.path}');
  print('File size: ${file.lengthSync()} bytes');
  print('Last modified: ${file.lastModifiedSync()}');
}
```

**Available methods:**
- `.path` - Returns relative file path
- `.absolute.path` - Returns full filesystem path
- `.lengthSync()` - Returns file size in bytes
- `.lastModifiedSync()` - Returns last modification timestamp

⚠️ **Note:** Attempting to access information for non-existent files throws exceptions.

## Reading CSV Files

### CSV File Overview

"A CSV (Comma Separated Values) file is a plain text file that contains data organized in a table format, where columns are separated by commas and rows are separated by line breaks."

**Common CSV use cases:**
- Data exchange between applications
- Data backup and restoration
- Database import/export operations
- Automating data processing workflows

### Parsing CSV Content

```dart
import 'dart:io';

void main() {
  File file = File('test.csv');
  String contents = file.readAsStringSync();

  List<String> lines = contents.split('\n');

  for (var line in lines) {
    print(line);
  }
}
```

Use the `split()` method to separate the file content by newlines, creating individual line records for processing.

## Partial File Reading

Extract specific portions of file content using `substring()`:

```dart
import 'dart:io';

void main() {
  File file = File('test.txt');
  String contents = file.readAsStringSync().substring(0, 10);
  print(contents);
}
```

This example retrieves only the first ten characters from the file.

## Reading from Specific Directories

Provide full file paths to access files outside the current working directory:

```dart
import 'dart:io';

void main() {
  File file = File('C:\\Users\\test.txt');
  String contents = file.readAsStringSync();
  print(contents);
}
```

Use absolute paths with proper directory separators for your operating system (backslashes for Windows, forward slashes for Unix-like systems).

---

## Source

- **URL**: https://dart-tutorial.com/file-handling-in-dart/read-file-in-dart/
- **Fetched**: 2026-01-27
