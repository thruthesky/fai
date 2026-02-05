# User Input in Dart

## Overview

The page explains how to accept user input in Dart, making programs more interactive and dynamic rather than relying on hard-coded values.

**Important Requirement:** To implement user input functionality, you must include `import 'dart:io';` at the beginning of your program.

**Note on Limitations:** "You won't be able to take input from users using dartpad. You need to run a program from your computer."

## String User Input

String inputs store textual data from users, such as names, addresses, or descriptions.

```dart
import 'dart:io';

void main() {
  print("Enter name:");
  String? name  = stdin.readLineSync();
  print("The entered name is ${name}");
}
```

**Sample Output:**
```
Enter name:
Raj Sharma
The entered name is Raj Sharma
```

## Integer User Input

Integer inputs retrieve whole numbers from users (examples: 10, 100, -800).

```dart
import 'dart:io';

void main() {
  print("Enter number:");
  int? number = int.parse(stdin.readLineSync()!);
  print("The entered number is ${number}");
}
```

**Sample Output:**
```
Enter number:
50
The entered number is 50
```

## Floating Point User Input

Double/float inputs capture decimal numbers from users (examples: 10.5, 100.5, -800.9).

```dart
import 'dart:io';

void main() {
  print("Enter a floating number:");
  double number = double.parse(stdin.readLineSync()!);
  print("The entered num is $number");
}
```

**Sample Output:**
```
Enter a floating number:
55.5
The entered num is 55.5
```

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/user-input-in-dart/
- **Fetched**: 2026-01-27
