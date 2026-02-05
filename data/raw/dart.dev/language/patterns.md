# Patterns in Dart

## Overview

Patterns represent the shape of values in Dart code. Per the documentation, "A pattern represents the shape of a set of values that it may match against actual values." They enable matching and destructuring operations throughout the language.

**Version requirement:** Patterns require a language version of at least 3.0.

## What Patterns Do

Patterns perform two primary functions:

### Matching

Pattern matching tests whether a value has an expected form. The specifics depend on the pattern type. For instance, a constant pattern succeeds when the value equals the constant:

```dart
switch (number) {
  case 1:
    print('one');
}
```

Patterns support recursive matching through subpatterns. In collection patterns, individual fields can be variable or constant patterns:

```dart
const a = 'a';
const b = 'b';
switch (obj) {
  case [a, b]:
    print('$a, $b');
}
```

The wildcard pattern (`_`) ignores matched values, while rest elements handle multiple items in list patterns.

### Destructuring

Destructuring extracts components from matched objects and binds them to new variables. This simplifies working with complex data structures:

```dart
var numList = [1, 2, 3];
var [a, b, c] = numList;
print(a + b + c);
```

Patterns can be nested arbitrarily:

```dart
switch (list) {
  case ['a' || 'b', var c]:
    print(c);
}
```

## Where Patterns Appear

Dart permits patterns in:

- Local variable declarations and assignments
- For and for-in loops
- If-case and switch-case statements
- Control flow in collection literals

### Variable Declaration

Pattern variable declarations begin with `var` or `final`, followed by the pattern:

```dart
var (a, [b, c]) = ('str', [1, 2]);
```

### Variable Assignment

Assignment patterns fall on the left side of assignments, working with existing variables rather than creating new ones. A practical use is swapping values:

```dart
var (a, b) = ('left', 'right');
(b, a) = (a, b);
print('$a $b'); // Prints "right left"
```

### Switch Statements and Expressions

Case clauses contain patterns that are refutable—they either match and destructure or continue execution. Variables bound in cases are scoped to that case body:

```dart
switch (obj) {
  case 1:
    print('one');
  case >= first && <= last:
    print('in range');
  case (var a, var b):
    print('a = $a, b = $b');
  default:
}
```

Logical-or patterns enable multiple cases to share behavior:

```dart
var isPrimary = switch (color) {
  Color.red || Color.yellow || Color.blue => true,
  _ => false,
};
```

Guard clauses evaluate conditions within cases without exiting on failure:

```dart
switch (pair) {
  case (int a, int b) when a > b:
    print('First element greater');
  case (int a, int b):
    print('First element not greater');
}
```

### For and For-in Loops

Patterns destructure collection elements during iteration. Object destructuring in for-in loops is particularly useful:

```dart
Map<String, int> hist = {'a': 23, 'b': 100};

for (var MapEntry(key: key, value: count) in hist.entries) {
  print('$key occurred $count times');
}
```

Shorthand notation allows inferring getter names from variable names:

```dart
for (var MapEntry(:key, value: count) in hist.entries) {
  print('$key occurred $count times');
}
```

## Common Use Cases

### Destructuring Multiple Returns

Records combined with patterns provide clean syntax for handling functions returning multiple values:

```dart
// Instead of:
var info = userInfo(json);
var name = info.$1;
var age = info.$2;

// Use:
var (name, age) = userInfo(json);
```

Named record fields work similarly:

```dart
final (:name, :age) = getData();
```

### Destructuring Class Instances

Object patterns match against named types and destructure their exposed getters:

```dart
final Foo myFoo = Foo(one: 'one', two: 2);
var Foo(:one, :two) = myFoo;
print('one $one, two $two');
```

### Algebraic Data Types

This approach applies operations to related type families by switching over subtypes rather than spreading behavior across all implementations:

```dart
sealed class Shape {}

class Square implements Shape {
  final double length;
  Square(this.length);
}

class Circle implements Shape {
  final double radius;
  Circle(this.radius);
}

double calculateArea(Shape shape) => switch (shape) {
  Square(length: var l) => l * l,
  Circle(radius: var r) => math.pi * r * r,
};
```

### Validating Incoming JSON

Map and list patterns validate structured data concisely. Instead of nested conditionals checking types and structure:

```dart
if (data is Map<String, Object?> &&
    data.length == 1 &&
    data.containsKey('user')) {
  var user = data['user'];
  if (user is List<Object> &&
      user.length == 2 &&
      user[0] is String &&
      user[1] is int) {
    var name = user[0] as String;
    var age = user[1] as int;
    print('User $name is $age years old.');
  }
}
```

A single if-case pattern handles it elegantly:

```dart
if (data case {'user': [String name, int age]}) {
  print('User $name is $age years old.');
}
```

This validates the map structure, confirms the 'user' key exists, verifies the paired value is a two-element list, checks element types, and binds variables—all in one expression.
