# Methods in Dart

Methods are functions that provide behavior for an object.

## Instance Methods

Instance methods on objects can access instance variables and `this`. Here's an example:

```dart
import 'dart:math';

class Point {
  final double x;
  final double y;

  Point(this.x, this.y);

  double distanceTo(Point other) {
    var dx = x - other.x;
    var dy = y - other.y;
    return sqrt(dx * dx + dy * dy);
  }
}
```

## Operators

Dart allows you to define operators with special names. The supported operator names include:

`<`, `>`, `<=`, `>=`, `==`, `~`, `-`, `+`, `/`, `~/`, `*`, `%`, `|`, `^`, `&`, `<<`, `>>>`, `>>`, `[]=`, `[]`

Here's an example defining vector operations:

```dart
class Vector {
  final int x, y;

  Vector(this.x, this.y);

  Vector operator +(Vector v) => Vector(x + v.x, y + v.y);
  Vector operator -(Vector v) => Vector(x - v.x, y - v.y);

  @override
  bool operator ==(Object other) =>
      other is Vector && x == other.x && y == other.y;

  @override
  int get hashCode => Object.hash(x, y);
}
```

## Getters and Setters

Getters and setters provide read and write access to object properties using `get` and `set` keywords:

```dart
class Rectangle {
  double left, top, width, height;

  Rectangle(this.left, this.top, this.width, this.height);

  double get right => left + width;
  set right(double value) => left = value - width;
  double get bottom => top + height;
  set bottom(double value) => top = value - height;
}

void main() {
  var rect = Rectangle(3, 4, 20, 15);
  assert(rect.left == 3);
  rect.right = 12;
  assert(rect.left == -8);
}
```

This approach allows starting with instance variables and wrapping them with methods later without changing client code.

## Abstract Methods

Instance, getter, and setter methods can be abstract, defining an interface while leaving implementation to subclasses. Abstract methods exist only in abstract classes or mixins and use a semicolon instead of a body:

```dart
abstract class Doer {
  void doSomething();
}

class EffectiveDoer extends Doer {
  void doSomething() {
    // Implementation here
  }
}
```
