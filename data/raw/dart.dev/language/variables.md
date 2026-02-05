# Dart Variables Guide

## Overview

Variables in Dart store references to objects. The language uses type inference, allowing you to declare variables with `var`, explicitly specify types, or use `Object` for flexible typing.

```dart
var name = 'Bob';
String name = 'Bob';
Object name = 'Bob';
```

## Null Safety

Dart enforces sound null safety, preventing null dereference errors at compile time rather than runtime.

### Key Features:

1. **Nullable types** use the `?` modifier to allow `null` values:
```dart
String? name     // Can be null or string
String name      // Cannot be null
```

2. **Non-nullable variables** must be initialized before use, while nullable ones default to `null`:
```dart
int? lineCount;
assert(lineCount == null);

int lineCount = 0;  // Required initialization
```

3. **Property access restrictions** prevent calling methods on potentially null expressions unless `null` supports them.

## Default Values

Uninitialized nullable variables default to `null`, including numeric types. Non-nullable variables must have explicit initialization before use. Dart's control flow analysis verifies this requirement.

```dart
int lineCount;

if (weLikeToCount) {
  lineCount = countLines();
} else {
  lineCount = 0;
}

print(lineCount);  // Valid - definitely assigned
```

## Late Variables

The `late` modifier serves two purposes:

1. **Declaring non-nullable variables** initialized after declaration
2. **Lazy initialization** of expensive operations

```dart
late String description;

void main() {
  description = 'Feijoada!';
  print(description);
}

late String temperature = readThermometer();  // Executes on first use
```

Runtime errors occur if late variables are accessed before initialization.

## Final and Const

**Final variables** can be assigned only once, while **const variables** are compile-time constants (implicitly final).

```dart
final name = 'Bob';
final String nickname = 'Bobby';

name = 'Alice';  // Error

const bar = 1000000;
const double atm = 1.01325 * bar;

baz = [42];  // Error - const cannot be reassigned
```

Const values can be created explicitly:
```dart
var foo = const [];
final bar = const [];
const baz = [];  // Equivalent to const []
```

Const declarations support type checks, casts, collection conditionals, and spread operators.

## Wildcard Variables

Introduced in language version 3.7, wildcard variables use the `_` identifier as non-binding placeholders. Initializers execute but values remain inaccessible.

Valid contexts:
- Local variable declarations
- For loop variables
- Catch clause parameters
- Generic type parameters
- Function parameters

```dart
main() {
  var _ = 1;
  int _ = 2;
}

for (var _ in list) {}

try {
  throw '!';
} catch (_) {
  print('oops');
}
```

The `unnecessary_underscores` lint identifies opportunities to replace multiple binding underscores with a single wildcard.

---

**Last Updated:** November 13, 2025 | **Dart Version:** 3.10.3
