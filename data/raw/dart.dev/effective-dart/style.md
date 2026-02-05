# Effective Dart: Style Guide

## Overview

This guide establishes conventions for consistent, readable Dart code through proper formatting and naming conventions. "A surprisingly important part of good code is good style."

## Identifiers

Dart uses three naming conventions:

- **UpperCamelCase**: Capitalize first letter of each word
- **lowerCamelCase**: Capitalize first letter of each word except the first
- **lowercase_with_underscores**: Only lowercase letters with underscores between words

### Type Names

Classes, enums, typedefs, and type parameters should use UpperCamelCase:

```dart
class SliderMenu { }
class HttpRequest { }
typedef Predicate<T> = bool Function(T value);
```

### Extension Names

Extensions use UpperCamelCase like regular types:

```dart
extension MyFancyList<T> on List<T> { }
extension SmartIterable<T> on Iterable<T> { }
```

### File and Package Names

Use lowercase_with_underscores for packages, directories, and source files:

```
my_package
└─ lib
   └─ file_system.dart
   └─ slider_menu.dart
```

### Import Prefixes

Import aliases should use lowercase_with_underscores:

```dart
import 'dart:math' as math;
import 'package:angular_components/angular_components.dart' as angular_components;
```

### Other Identifiers

Variables, parameters, and class members use lowerCamelCase:

```dart
var count = 3;
HttpRequest httpRequest;
void align(bool clearItems) { }
```

### Constants

New code should use lowerCamelCase for constants and enum values:

```dart
const pi = 3.14;
const defaultTimeout = 1000;
final urlScheme = RegExp('^([a-z]+):');
```

### Acronyms

Capitalize acronyms longer than two letters like regular words:

```dart
// Longer than two letters:
Http, Nasa, Uri, Esq, Ave

// Two letters, capitalized in English:
ID, TV, UI
```

## Import Ordering

### Dart Imports First

Place dart: imports before other imports:

```dart
import 'dart:async';
import 'dart:collection';

import 'package:bar/bar.dart';
import 'package:foo/foo.dart';
```

### Package Before Relative

Place package: imports before relative imports.

### Alphabetical Sorting

Sort each section alphabetically.

## Formatting

### Use dart format

The official whitespace-handling rules for Dart are whatever dart format produces.

### Line Length Preference

Prefer lines of 80 characters or fewer.

### Curly Braces

Always use curly braces for flow control statements to avoid the dangling else problem:

```dart
if (isWeekDay) {
  print('Bike to work!');
} else {
  print('Go dancing or read a book!');
}
```

---

## Source

- **URL**: https://dart.dev/effective-dart/style
- **Fetched**: 2026-01-27
