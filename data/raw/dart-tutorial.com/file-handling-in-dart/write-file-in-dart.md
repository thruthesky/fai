# Writing Files in Dart - Complete Guide

## Introduction

This tutorial covers file writing operations in Dart using the `File` class and related methods from the `dart:io` library.

## Basic File Writing

### Method: writeAsStringSync()

The fundamental approach to writing files involves creating a `File` object and using the synchronous write method.

**Example:**
```dart
import 'dart:io';

void main() {
  File file = File('test.txt');
  file.writeAsStringSync('Welcome to test.txt file.');
  print('File written.');
}
```

**Output:**
```
File written.
```

### Important Consideration

"If you have already some content in **test.txt** file, then it will be" overwritten with the new data. This destructive behavior is the default behavior when writing without append mode.

## Appending Content to Existing Files

### Using FileMode.append

To preserve existing content while adding new data, utilize the append mode parameter.

**Example:**
```dart
import 'dart:io';

void main() {
  File file = File('test.txt');
  file.writeAsStringSync('\nThis is a new content.',
                         mode: FileMode.append);
  print('Congratulations!! New content is added on top of previous content.');
}
```

**Scenario:** When `test.txt` already contains `Welcome to test.txt file.`, this code adds a new line without removing existing data.

## Creating CSV Files

### Interactive CSV Writer Example

This example demonstrates gathering user input and writing structured data:

```dart
import 'dart:io';

void main() {
  File file = File("students.csv");
  file.writeAsStringSync('Name,Phone\n');

  for (int i = 0; i < 3; i++) {
    stdout.write("Enter name of student ${i + 1}: ");
    String? name = stdin.readLineSync();
    stdout.write("Enter phone of student ${i + 1}: ");
    String? phone = stdin.readLineSync();
    file.writeAsStringSync('$name,$phone\n',
                           mode: FileMode.append);
  }
  print("Congratulations!! CSV file written successfully.");
}
```

**Sample Output:**
```
Name,Phone
John,1234567890
Mark,0123456789
Elon,0122112322
```

## Versatility

The `writeAsStringSync()` method supports creating various file types beyond plain text, including `.html`, `.json`, and `.xml` formats.

---

## Source

- **URL**: https://dart-tutorial.com/file-handling-in-dart/write-file-in-dart/
- **Fetched**: 2026-01-27
