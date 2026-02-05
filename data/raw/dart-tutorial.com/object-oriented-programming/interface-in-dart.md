# Interfaces in Dart

## Definition

"An interface defines a syntax that a class must follow." It functions as a contract establishing class capabilities and is used to achieve abstraction. When implementing an interface, all its properties and methods must be implemented.

## Key Characteristics

- Dart has no dedicated `interface` keyword
- Every class implicitly defines an interface
- Use `abstract class` to declare interfaces (most common approach)
- Use the `implements` keyword to implement an interface

## Basic Syntax

```dart
class InterfaceName {
  // code
}

class ClassName implements InterfaceName {
  // code
}
```

## Example: Vehicle Interface

```dart
abstract class Vehicle {
  void start();
  void stop();
}

class Car implements Vehicle {
  @override
  void start() {
    print('Car started');
  }

  @override
  void stop() {
    print('Car stopped');
  }
}

void main() {
  var car = Car();
  car.start();
  car.stop();
}
```

Output:
```
Car started
Car stopped
```

## Multiple Interface Implementation

A class can implement multiple interfaces simultaneously:

```dart
class ClassName implements Interface1, Interface2, Interface3 {
  // code
}
```

### Example: Multiple Interfaces

```dart
abstract class Area {
  void area();
}

abstract class Perimeter {
  void perimeter();
}

class Rectangle implements Area, Perimeter {
  int length, breadth;

  Rectangle(this.length, this.breadth);

  @override
  void area() {
    print('The area of the rectangle is ${length * breadth}');
  }

  @override
  void perimeter() {
    print('The perimeter of the rectangle is ${2 * (length + breadth)}');
  }
}

void main() {
  Rectangle rectangle = Rectangle(10, 20);
  rectangle.area();
  rectangle.perimeter();
}
```

Output:
```
The area of the rectangle is 200
The perimeter of the rectangle is 60
```

## Extends vs. Implements

| Feature | Extends | Implements |
|---------|---------|------------|
| Purpose | Inherit a class | Inherit as interface |
| Method Definition | Complete | Abstract |
| Multiple Usage | One class only | Multiple classes |
| Override Requirement | Optional | Required |
| Constructor Behavior | Called before subclass | Not called |
| Super Keyword | Supported | Not supported |
| Field Override | Not required | Required |

## Key Takeaways

- Interfaces define contracts for class behavior
- Use `abstract class` declarations for interfaces
- Single inheritance allowed; multiple interface implementation possible
- Achieve multiple inheritance through interface implementation
- All interface methods must be overridden in implementing classes

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/interface-in-dart/
- **Fetched**: 2026-01-27
