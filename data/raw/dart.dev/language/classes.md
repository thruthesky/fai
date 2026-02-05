# Dart Classes: Complete Page Content

## Overview

"Dart is an object-oriented language with classes and mixin-based inheritance." Every object derives from the `Object` class (except `Null`). Mixin-based inheritance allows class bodies to be reused across multiple hierarchies. Extension methods add functionality without modifying classes, while class modifiers control subtyping permissions.

## Using Class Members

Objects contain members—methods and instance variables. Access them using dot notation:

```dart
var p = Point(2, 2);
assert(p.y == 2);
double distance = p.distanceTo(Point(4, 4));
```

For potentially null objects, use the safe navigation operator:

```dart
var a = p?.y;
```

## Using Constructors

Constructors create objects. Names follow the pattern `ClassName` or `ClassName.identifier`:

```dart
var p1 = Point(2, 2);
var p2 = Point.fromJson({'x': 1, 'y': 2});
```

The `new` keyword is optional. For constant constructors, use `const`:

```dart
var p = const ImmutablePoint(2, 2);
```

Identical compile-time constants reference the same instance:

```dart
var a = const ImmutablePoint(1, 1);
var b = const ImmutablePoint(1, 1);
assert(identical(a, b)); // Same instance
```

Within constant contexts, `const` can be omitted after the first use:

```dart
const pointAndLine = {
  'point': [ImmutablePoint(0, 0)],
  'line': [ImmutablePoint(1, 10), ImmutablePoint(-2, 11)],
};
```

Non-constant constructors outside constant contexts create non-constant objects:

```dart
var a = const ImmutablePoint(1, 1); // Constant
var b = ImmutablePoint(1, 1);       // Non-constant
assert(!identical(a, b));
```

## Getting an Object's Type

Retrieve runtime type information using `runtimeType`:

```dart
print('The type of a is ${a.runtimeType}');
```

Prefer type test operators (`is`) over `runtimeType` comparisons for production code stability.

## Instance Variables

Declare instance variables within classes:

```dart
class Point {
  double? x; // Initially null
  double? y; // Initially null
  double z = 0; // Initially 0
}
```

Nullable uninitialized variables default to `null`. Non-nullable variables require initialization.

Instance variables automatically generate getter methods. Non-final variables and `late final` variables without initializers generate setter methods:

```dart
class Point {
  double? x;
  double? y;
}

void main() {
  var point = Point();
  point.x = 4; // Uses setter
  assert(point.x == 4); // Uses getter
  assert(point.y == null);
}
```

Non-`late` instance variable initializers cannot access `this`:

```dart
double initialX = 1.5;

class Point {
  double? x = initialX;        // OK
  double? y = this.x;          // ERROR
  late double? z = this.x;     // OK
  Point(this.x, this.y);       // OK
}
```

Final instance variables must be set exactly once:

```dart
class ProfileMark {
  final String name;
  final DateTime start = DateTime.now();

  ProfileMark(this.name);
  ProfileMark.unnamed() : name = '';
}
```

For post-constructor assignment, use factory constructors or `late final` (with caveats).

## Implicit Interfaces

Every class implicitly defines an interface containing all instance members. Classes can explicitly implement interfaces:

```dart
class Person {
  final String _name;
  Person(this._name);
  String greet(String who) => 'Hello, $who. I am $_name.';
}

class Impostor implements Person {
  String get _name => '';
  String greet(String who) => 'Hi $who. Do you know who I am?';
}

String greetBob(Person person) => person.greet('Bob');

void main() {
  print(greetBob(Person('Kathy')));
  print(greetBob(Impostor()));
}
```

Multiple interfaces are specified with commas:

```dart
class Point implements Comparable, Location {
  ...
}
```

## Class Variables and Methods

The `static` keyword creates class-level variables and methods.

### Static Variables

Static variables maintain class-wide state and constants:

```dart
class Queue {
  static const initialCapacity = 16;
}

void main() {
  assert(Queue.initialCapacity == 16);
}
```

Static variables initialize only upon first use. The style guide recommends `lowerCamelCase` for constant names.

### Static Methods

Static methods operate at the class level without access to instance data via `this`:

```dart
import 'dart:math';

class Point {
  double x, y;
  Point(this.x, this.y);

  static double distanceBetween(Point a, Point b) {
    var dx = a.x - b.x;
    var dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
  }
}

void main() {
  var a = Point(2, 2);
  var b = Point(4, 4);
  var distance = Point.distanceBetween(a, b);
  assert(2.8 < distance && distance < 2.9);
  print(distance);
}
```

Consider top-level functions for common utilities instead of static methods. Static methods can serve as compile-time constants passed to constant constructors.
