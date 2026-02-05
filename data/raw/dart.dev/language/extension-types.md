# Extension Types in Dart

## Overview

Extension types are a compile-time abstraction that wraps an existing type with a different, static-only interface. They provide discipline on available operations without runtime overhead, making them ideal for static JS interop and type safety improvements.

Key distinction: Extension types differ from extension methods. While extension methods add functionality directly to instances, extension types create a separate interface that only applies when the static type matches the extension type.

## Syntax

### Declaration

Define an extension type using the `extension type` keyword with a representation type:

```dart
extension type E(int i) {
  // Define operations
}
```

The representation declaration includes:
- An implicit getter returning the representation type
- An implicit constructor
- A reference variable (`i` in this example) for accessing the underlying object

Generic extension types are supported:

```dart
extension type E<T>(List<T> elements) {
  // ...
}
```

### Constructors

Extension types support custom constructors. The representation declaration serves as an implicit unnamed constructor. Additional constructors must initialize the representation variable:

```dart
extension type E(int i) {
  E.n(this.i);
  E.m(int j, String foo) : i = j + foo.length;
}
```

You can hide the default constructor using private syntax:

```dart
extension type E._(int i) {
  E.fromString(String foo) : i = int.parse(foo);
}
```

### Members

Members are declared like class members. Allowed constructs include methods, getters, setters, and operators. Instance variables and abstract members are not permitted:

```dart
extension type NumberE(int value) {
  NumberE operator +(NumberE other) =>
      NumberE(value + other.value);

  NumberE get myNum => this;

  bool isValid() => !value.isNegative;
}
```

### Implements Clause

The `implements` keyword establishes subtype relationships and grants access to representation type members. Valid targets include:

**The representation type itself:**

```dart
extension type NumberI(int i) implements int {
  // All int members available
}
```

**Supertypes of the representation type:**

```dart
extension type Sequence<T>(List<T> _) implements Iterable<T> {
  // Better operations than List
}
```

**Other extension types on the same representation:**

```dart
extension type const Val<T>._(({T value}) _) implements Opt<T> {
  const Val(T value) : this._((value: value));
  T get value => _.value;
}
```

#### `@redeclare` Annotation

Members sharing names with supertype members constitute redeclaration, not override. Use the `@redeclare` annotation to signal intentional replacement:

```dart
import 'package:meta/meta.dart';

extension type MyString(String _) implements String {
  @redeclare
  int operator [](int index) => codeUnitAt(index);
}
```

## Usage

### Creating Instances

Instantiate extension types like classes:

```dart
extension type NumberE(int value) {
  NumberE operator +(NumberE other) =>
      NumberE(value + other.value);
}

void testE() {
  var num = NumberE(1);
}
```

### Use Case 1: Extended Interface (Transparent)

When implementing the representation type, the extension type becomes "transparent," exposing all representation type members alongside new ones:

```dart
extension type NumberT(int value) implements int {
  NumberT get i => this;
}

void main () {
  var v1 = NumberT(1);
  int v2 = NumberT(2);
  var v3 = v1.i - v1;  // OK: int member invocation
  var v4 = v2 + v1;    // OK: arithmetic operation
}
```

### Use Case 2: Different Interface (Opaque)

Extension types not implementing their representation type act as distinct types, hiding representation members:

```dart
extension type IdNumber(int id) {
  operator <(IdNumber other) => id < other.id;
}

void main() {
  var safeId = IdNumber(42424242);
  safeId + 10;           // Error: no '+' operator
  int myId = safeId;     // Error: wrong type
  myId = safeId as int;  // OK: explicit cast
}
```

## Type Considerations

**Critical limitation:** Extension types are purely compile-time constructs. At runtime, no trace of the extension type exists—only the representation type.

Runtime type operations evaluate against the representation type:

```dart
void main() {
  var n = NumberE(1);

  if (n is int) print(n);  // Prints 1

  if (n case int x) print(x.toRadixString(10));  // Prints 1

  switch (n) {
    case int(:var isEven): print("$n");  // Works as int
  }
}
```

Conversely, an `int` value can match an extension type pattern:

```dart
void main() {
  int i = 2;
  if (i is NumberE) print("It is");  // Matches
  if (i case NumberE v) print("value: ${v.value}");  // Works
}
```

**Key implication:** Unlike wrapper classes, extension types cannot fully encapsulate wrapped objects. The underlying representation remains accessible at runtime, making them "unsafe" abstractions in security-critical contexts. However, they provide zero-cost static discipline and improved performance compared to wrapper objects.

Generics are erased at runtime—`List<E>` is identical to `List<R>` at runtime, where `R` is the representation type.
