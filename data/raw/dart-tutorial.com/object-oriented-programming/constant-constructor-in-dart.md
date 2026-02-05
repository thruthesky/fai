# Constant Constructor in Dart

## Overview

A constant constructor creates objects whose values cannot be changed. It uses the `const` keyword and improves program performance.

## Rules for Declaration

According to the tutorial, there are three key requirements:

1. "All properties of the class must be final"
2. The constructor has no body
3. Classes with const constructors must be initialized using the `const` keyword

## Code Example 1: Basic Implementation

```dart
class Point {
  final int x;
  final int y;

  const Point(this.x, this.y);
}

void main() {
  Point p1 = const Point(1, 2);
  print("The p1 hash code is: ${p1.hashCode}");

  Point p2 = const Point(1, 2);
  print("The p2 hash code is: ${p2.hashCode}");

  Point p3 = Point(2, 2);
  print("The p3 hash code is: ${p3.hashCode}");
}
```

**Key Insight**: "p1 and p2 has the same hash code. This is because p1 and p2 are constant objects."

## Code Example 2: With Optional Parameters

```dart
class Student {
  final String? name;
  final int? age;
  final int? rollNumber;

  const Student({this.name, this.age, this.rollNumber});
}

void main() {
  const Student student = Student(
    name: "John",
    age: 20,
    rollNumber: 1
  );
  print("Name: ${student.name}");
  print("Age: ${student.age}");
}
```

## Code Example 3: Named Parameters

```dart
class Car {
  final String? name;
  final String? model;
  final int? price;

  const Car({this.name, this.model, this.price});
}

void main() {
  const Car car = Car(
    name: "BMW",
    model: "X5",
    price: 50000
  );
  print("Name: ${car.name}");
}
```

## Benefits

The primary advantage is that constant constructors "Improves the performance of the program."

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/constant-constructor-in-dart/
- **Fetched**: 2026-01-27
