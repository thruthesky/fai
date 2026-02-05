# Dart Standard Input Output Tutorial

## Overview

In Dart, you can receive user input through the console using the **`.readLineSync()`** function. To utilize this functionality, you must import the **`dart:io`** library from Dart's standard libraries.

## The Stdin Class

The `Stdin` class enables reading data from standard input in both synchronous and asynchronous modes. The `readLineSync()` method is commonly used for retrieving user input. For additional methods, refer to the [official Dart documentation](https://api.dart.dev/stable/2.7.2/dart-io/Stdin-class.html).

## Taking String Input

```dart
import 'dart:io';

void main() {
    print("Enter your name?");

    // Reading name with null safety
    String? name = stdin.readLineSync();

    // Printing the name
    print("Hello, $name! \nWelcome to GeeksforGeeks!!");
}
```

**Input:** `Geek`

**Output:**
```
Enter your name?
Hello, Geek!
Welcome to GeeksforGeeks!!
```

## Taking Integer Input

```dart
import 'dart:io';

void main() {
    print("Enter your favourite number:");

    // Scanning and parsing number
    int? n = int.parse(stdin.readLineSync()!);

    // Printing that number
    print("Your favourite number is $n");
}
```

**Input:** `01`

**Output:**
```
Enter your favourite number:
Your favourite number is 1
```

## Standard Output Methods

Two primary approaches exist for console output:

1. **`print()`** - Moves cursor to next line
2. **`stdout.write()`** - Keeps cursor on same line

```dart
import 'dart:io';

void main() {
    // Using print statement
    print("Welcome to GeeksforGeeks!");

    // Using stdout.write()
    stdout.write("Welcome to GeeksforGeeks!");
}
```

**Output:**
```
Welcome to GeeksforGeeks!
Welcome to GeeksforGeeks!
```

> **Key Distinction:** The `print()` statement brings the cursor to the next line while `stdout.write()` doesn't bring the cursor to the next line, it remains in the same line.

## Practical Example: Addition Program

```dart
import 'dart:io';

void main() {
    print("-----------GeeksForGeeks-----------");
    print("Enter first number");
    int? n1 = int.parse(stdin.readLineSync()!);

    print("Enter second number");
    int? n2 = int.parse(stdin.readLineSync()!);

    // Adding and printing result
    int sum = n1 + n2;
    print("Sum is $sum");
}
```

**Input:**
```
11
12
```

**Output:**
```
-----------GeeksForGeeks-----------
Enter first number
Enter second number
Sum is 23
```

## Summary

Dart uses `stdin.readLineSync()` from the `dart:io` library to handle user input as either strings or numbers. The `Stdin` class supports both synchronous and asynchronous operations. For output, employ `print()` for new-line formatting or `stdout.write()` for same-line printing, enabling streamlined console interactions and basic arithmetic operations.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-standard-input-output/
- **Fetched**: 2026-01-27
