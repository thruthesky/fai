# Dart Libraries Guide

## Overview

Dart is an open-source language developed by Google for both server-side and client-side development. It's an object-oriented language similar to Java, extensively used for creating single-page websites and web applications like Gmail.

## Importing Libraries

To use libraries in Dart, you must import them first. The `import` keyword makes library components available in the current file.

**Syntax:**
```dart
import 'dart:sync';
```

**Example with built-in library:**
```dart
import 'dart:math';

void main() {
  print("Square root of 25 is: ${sqrt(25)}");
}
```

**Output:**
```
Square root of 25 is: 5
```

For importing from external packages, use the `package:` directive:
```dart
import 'package:utilities/utilities.dart';
```

## Creating Custom Libraries

Custom libraries are user-defined and created in two steps:

### 1. Declaration

Declare the library name using the `library` keyword:
```dart
library my_lib;
```

### 2. Connection

Import from the same or different directory:
```dart
// Same directory
import 'library_name';

// Different directory
import 'dir/library_name';
```

**Example:**
```dart
library basic_calc;

import 'dart:math';

int add(int a, int b) {
  print("Add Method");
  return a + b;
}

int multiplication(int a, int b) {
  print("Multiplication Method");
  return a * b;
}

int subtraction(int a, int b) {
  print("Subtraction Method");
  return a - b;
}

int modulus(int a, int b) {
  print("Modulus Method");
  return a % b;
}
```

**Using the library:**
```dart
import 'basic.dart';

void main() {
  var a = 50;
  var b = 30;

  var sum = add(a, b);
  var mod = modulus(a, b);
  var mul = multiplication(a, b);
  var sub = subtraction(a, b);

  print("$a + $b = $sum");
  print("$a % $b = $mod");
  print("$a * $b = $mul");
  print("$a - $b = $sub");
}
```

**Output:**
```
Add Method
Multiplication Method
Subtraction Method
Modulus Method
50 + 30 = 80
50 % 30 = 20
50 * 30 = 1500
50 - 30 = 20
```

## Encapsulation

Encapsulation combines data and functions into a single unit (class). Use underscore (`_`) prefix to make library content private.

**Important Note:** Encapsulation in Dart takes place at library level instead of class-level, unlike other OOP languages.

**Syntax:**
```dart
_identifier
```

**Example:**
```dart
library cake;

class MainCake {
  // Non-private property
  List<String> randomPieceOfCakes = ['chocolate', 'butterscotch',
                                      'vanilla', 'strawberry'];

  // Private properties
  String _pieceOfCake1 = "chocolate";
  String pieceOfCake2 = "butterscotch";
}
```

**Using encapsulated library:**
```dart
import 'cake.dart';

void main() {
  MainCake newCake = new MainCake();

  // Non-private property
  print(newCake.randomPieceOfCakes);

  // Private property
  print(newCake._pieceOfCake1);

  // Non-private property
  print(newCake.pieceOfCake2);
}
```

## The `as` Keyword

When importing multiple libraries with conflicting function names, use `as` to create library aliases.

**Syntax:**
```dart
import 'my_lib' as prefix;
```

**Example:**

Library 1:
```dart
library greetings;

void sayHi(msg) {
  print("Hello coder. Welcome to ${msg}");
}
```

Library 2:
```dart
library hellogreetings;

void sayHi(msg) {
  print("${msg} has solutions of all your problems");
}
```

**Using both libraries:**
```dart
import 'greetings.dart';
import 'hellogreetings.dart' as gret;

void main() {
  sayHi("GFG");        // First library
  gret.sayHi("GFG");   // Second library with alias
}
```

**Output:**
```
Hello coder. Welcome to GFG
GFG has solutions of all your problems
```

## Core Dart Libraries

### Multi-Platform Libraries

| Library | Purpose |
|---------|---------|
| `dart:async` | Asynchronous programming with Future and Stream classes |
| `dart:collection` | Collection classes and utilities |
| `dart:convert` | Encoders/decoders for JSON and UTF-8 |
| `dart:core` | Built-in types and core functionality |
| `dart:developer` | Developer tools interaction (debugger, inspector) |
| `dart:math` | Mathematical constants, functions, random number generation |
| `dart:typed_data` | Efficient handling of fixed-size data and SIMD types |

### Native Platform Libraries

| Library | Purpose |
|---------|---------|
| `dart:io` | File, socket, HTTP, and I/O support for non-web apps |
| `dart:isolate` | Concurrent programming with independent workers |

### Web Platform Libraries

| Library | Purpose |
|---------|---------|
| `dart:html` | HTML elements for web-based applications |
| `dart:indexed_db` | Client-side key-value store with index support |
| `dart:web_audio` | High-fidelity audio programming in browsers |
| `dart:web_gl` | 3D programming in browsers |

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-libraries/
- **Fetched**: 2026-01-27
