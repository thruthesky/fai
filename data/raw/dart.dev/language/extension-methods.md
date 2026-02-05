# Extension Methods in Dart

## Overview

Extension methods allow you to add functionality to existing libraries without modifying the original code. As described in the Dart documentation, "Extension methods add functionality to existing libraries. You might use extension methods without even knowing it."

When working with third-party APIs or widely-used libraries, direct modification is often impractical. Extensions solve this by enabling new capabilities on existing types.

### Basic Example

Instead of:
```dart
int.parse('42')
```

You can write:
```dart
'42'.parseInt()
```

This requires importing a library containing the extension:

```dart
import 'string_apis.dart';

void main() {
  print('42'.parseInt()); // Uses extension method
}
```

## Implementing Extensions

### Basic Syntax

```dart
extension NumberParsing on String {
  int parseInt() {
    return int.parse(this);
  }

  double parseDouble() {
    return double.parse(this);
  }
}
```

Extensions can contain methods, getters, setters, operators, static fields, and static helper methods.

### Unnamed Extensions

Extensions can omit their name, making them visible only within their declaration library:

```dart
extension on String {
  bool get isBlank => trim().isEmpty;
}
```

Unnamed extensions cannot be explicitly applied to resolve conflicts.

## Using Extension Methods

### Import and Usage

```dart
import 'string_apis.dart';

void main() {
  print('42'.padLeft(5));    // Standard method
  print('42'.parseInt());     // Extension method
}
```

### Static Types vs. Dynamic

Extension methods require static type information. They **do not work** with variables typed as `dynamic`:

```dart
dynamic d = '2';
print(d.parseInt()); // Runtime exception: NoSuchMethodError

var v = '2';
print(v.parseInt()); // Works fine - v inferred as String
```

This constraint exists because "extension methods are resolved against the static type of the receiver."

### Resolving API Conflicts

When multiple extensions define the same method name, use these strategies:

**1. Import Limiting:**
```dart
import 'string_apis.dart';
import 'string_apis_2.dart' hide NumberParsing2;

void main() {
  print('42'.parseInt()); // Uses first extension
}
```

**2. Explicit Application:**
```dart
import 'string_apis.dart';
import 'string_apis_2.dart';

void main() {
  print(NumberParsing('42').parseInt());
  print(NumberParsing2('42').parseInt());
}
```

**3. Import Prefixes:**
```dart
import 'string_apis.dart';
import 'string_apis_3.dart' as rad;

void main() {
  print(NumberParsing('42').parseInt());
  print(rad.NumberParsing('42').parseInt());
}
```

## Generic Extensions

Extensions support type parameters:

```dart
extension MyFancyList<T> on List<T> {
  int get doubleLength => length * 2;
  List<T> operator -() => reversed.toList();
  List<List<T>> split(int at) => [sublist(0, at), sublist(at)];
}
```

The type parameter `T` binds based on the static type of the receiver.

## Key Takeaways

- Extensions enable adding methods to existing types without inheritance
- They resolve statically, making them performant
- They work with type inference but not with `dynamic` types
- Named extensions help resolve naming conflicts
- Generic extensions provide flexible, type-safe extensions across parameterized types
