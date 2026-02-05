# Dart Constructors: Complete Guide

## Overview

"Constructors are special functions that create instances of classes." Dart supports multiple constructor types, each serving distinct purposes in object-oriented programming.

## Types of Constructors

### Generative Constructors

Standard constructors that instantiate classes and initialize instance variables:

```dart
class Point {
  double x;
  double y;

  Point(this.x, this.y);
}
```

### Default Constructors

When no constructor is explicitly defined, Dart automatically provides a default generative constructor with no arguments or name.

### Named Constructors

Enable multiple constructors per class with clarifying names:

```dart
class Point {
  final double x;
  final double y;

  Point(this.x, this.y);
  Point.origin() : x = 0, y = 0;
}
```

**Important note:** "A subclass doesn't inherit a superclass's named constructor."

### Constant Constructors

Create compile-time constant objects when all instance variables are final:

```dart
class ImmutablePoint {
  static const ImmutablePoint origin = ImmutablePoint(0, 0);
  final double x, y;

  const ImmutablePoint(this.x, this.y);
}
```

### Redirecting Constructors

Forward calls to another constructor within the same class using `this`:

```dart
class Point {
  double x, y;

  Point(this.x, this.y);
  Point.alongXAxis(double x) : this(x, 0);
}
```

### Factory Constructors

Used when constructors don't always create new instances or require non-trivial initialization:

```dart
class Logger {
  final String name;
  bool mute = false;
  static final Map<String, Logger> _cache = {};

  factory Logger(String name) {
    return _cache.putIfAbsent(name,
        () => Logger._internal(name));
  }

  factory Logger.fromJson(Map<String, Object> json) {
    return Logger(json['name'].toString());
  }

  Logger._internal(this.name);
}
```

**Warning:** "Factory constructors can't access `this`."

### Redirecting Factory Constructors

Delegates to another class's constructor:

```dart
factory Listenable.merge(List<Listenable> listenables) =
    _MergingListenable
```

### Constructor Tear-Offs

Supplies constructors as parameters without invoking them:

```dart
// Good: Use tear-offs
var strings = charCodes.map(String.fromCharCode);
var buffers = charCodes.map(StringBuffer.new);

// Avoid: Lambda wrappers
var strings = charCodes.map((code) => String.fromCharCode(code));
```

## Instance Variable Initialization

### Declaration Initialization

Set default values at variable declaration:

```dart
class PointA {
  double x = 1.0;
  double y = 2.0;
}
```

### Initializing Formal Parameters

Simplify assignment from constructor arguments:

```dart
class PointB {
  final double x;
  final double y;

  PointB(this.x, this.y);
  PointB.optional([this.x = 0.0, this.y = 0.0]);
}
```

Works with named parameters:

```dart
class PointC {
  double x;
  double y;

  PointC.named({this.x = 1.0, this.y = 1.0});
}
```

### Initializer Lists

Initialize instance variables before the constructor body executes:

```dart
Point.fromJson(Map<String, double> json)
    : x = json['x']!,
      y = json['y']! {
  print('In Point.fromJson(): ($x, $y)');
}
```

Validate inputs using assertions:

```dart
Point.withAssert(this.x, this.y) : assert(x >= 0) {
  print('In Point.withAssert(): ($x, $y)');
}
```

**Warning:** "The right-hand side of an initializer list can't access `this`."

## Constructor Inheritance

### Superclass Constructors

"Subclasses, or child classes, don't inherit constructors from their superclass." If a superclass lacks an unnamed, no-argument constructor, explicitly call a superclass constructor:

```dart
class Person {
  Person.fromJson(Map data) {
    print('in Person');
  }
}

class Employee extends Person {
  Employee.fromJson(Map data) : super.fromJson(data) {
    print('in Employee');
  }
}
```

Constructor execution order: initializer list → superclass constructor → main class constructor.

### Super Parameters

Forward parameters to superclass constructors (requires language version 2.17+):

```dart
class Vector2d {
  final double x;
  final double y;

  Vector2d(this.x, this.y);
}

class Vector3d extends Vector2d {
  final double z;

  Vector3d(super.x, super.y, this.z);
}
```

For named superclass constructors:

```dart
class Vector3d extends Vector2d {
  final double z;

  Vector3d.yzPlane({required super.y, required this.z})
      : super.named(x: 0);
}
```
