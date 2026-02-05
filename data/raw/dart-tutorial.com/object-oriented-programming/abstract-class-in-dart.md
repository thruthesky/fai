# Abstract Class in Dart

## Overview

"Abstract classes are classes that cannot be initialized. It is used to define the behavior of a class that can be inherited by other classes."

Abstract classes serve as blueprints for subclasses and are declared using the `abstract` keyword.

## Syntax

```dart
abstract class ClassName {
  //Body of abstract class

  method1();
  method2();
}
```

## Abstract Methods

Abstract methods lack implementation details and are declared with only a semicolon instead of a method body.

```dart
abstract class ClassName {
  //Body of abstract class
  method1();
  method2();
}
```

## Why Use Abstract Classes

"Subclasses of an abstract class must implement all the abstract methods of the abstract class." This enforces a contract where derived classes must provide concrete implementations, enabling abstraction throughout the codebase.

## Example 1: Vehicle Implementation

```dart
abstract class Vehicle {
  void start();
  void stop();
}

class Car extends Vehicle {
  @override
  void start() {
    print('Car started');
  }

  @override
  void stop() {
    print('Car stopped');
  }
}

class Bike extends Vehicle {
  @override
  void start() {
    print('Bike started');
  }

  @override
  void stop() {
    print('Bike stopped');
  }
}

void main() {
  Car car = Car();
  car.start();
  car.stop();

  Bike bike = Bike();
  bike.start();
  bike.stop();
}
```

**Output:**
```
Car started
Car stopped
Bike started
Bike stopped
```

## Example 2: Shape Calculation

```dart
abstract class Shape {
  int dim1, dim2;

  Shape(this.dim1, this.dim2);

  void area();
}

class Rectangle extends Shape {
  Rectangle(int dim1, int dim2) : super(dim1, dim2);

  @override
  void area() {
    print('The area of the rectangle is ${dim1 * dim2}');
  }
}

class Triangle extends Shape {
  Triangle(int dim1, int dim2) : super(dim1, dim2);

  @override
  void area() {
    print('The area of the triangle is ${0.5 * dim1 * dim2}');
  }
}

void main() {
  Rectangle rectangle = Rectangle(10, 20);
  rectangle.area();

  Triangle triangle = Triangle(10, 20);
  triangle.area();
}
```

**Output:**
```
The area of the rectangle is 200
The area of the triangle is 100.0
```

## Constructors in Abstract Classes

While you cannot instantiate abstract classes directly, "you can define a constructor in an abstract class. The constructor of an abstract class is called when an object of a subclass is created."

## Example 3: Bank Interest Rates

```dart
abstract class Bank {
  String name;
  double rate;

  Bank(this.name, this.rate);

  void interest();

  void display() {
    print('Bank Name: $name');
  }
}

class SBI extends Bank {
  SBI(String name, double rate) : super(name, rate);

  @override
  void interest() {
    print('The rate of interest of SBI is $rate');
  }
}

class ICICI extends Bank {
  ICICI(String name, double rate) : super(name, rate);

  @override
  void interest() {
    print('The rate of interest of ICICI is $rate');
  }
}

void main() {
  SBI sbi = SBI('SBI', 8.4);
  ICICI icici = ICICI('ICICI', 7.3);

  sbi.interest();
  icici.interest();
  icici.display();
}
```

**Output:**
```
The rate of interest of SBI is 8.4
The rate of interest of ICICI is 7.3
Bank Name: ICICI
```

## Key Points to Remember

- You cannot instantiate abstract classes
- They support both abstract and non-abstract methods
- They define behavior inherited by subclasses
- Abstract methods contain only signatures without implementations

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/abstract-class-in-dart/
- **Fetched**: 2026-01-27
