# Dart Classes: Complete Guide

## Overview

In Dart's object-oriented programming model, "a class is a blueprint for creating objects. A class defines the properties and methods that an object will have."

## Class Declaration Syntax

```dart
class ClassName {
  // properties or fields
  // methods or functions
}
```

**Key requirements:**
- Use the `class` keyword to define a class
- Class names must start with a capital letter (PascalCase convention)
- Class body contains properties and functions
- Properties store data (also called fields or attributes)
- Functions perform operations (also called methods)

## Practical Examples

### Example 1: Animal Class

```dart
class Animal {
  String? name;
  int? numberOfLegs;
  int? lifeSpan;

  void display() {
    print("Animal name: $name.");
    print("Number of Legs: $numberOfLegs.");
    print("Life Span: $lifeSpan.");
  }
}
```

### Example 2: Person Class

```dart
class Person {
  String? name;
  String? phone;
  bool? isMarried;
  int? age;

  void displayInfo() {
    print("Person name: $name.");
    print("Phone number: $phone.");
    print("Married: $isMarried.");
    print("Age: $age.");
  }
}
```

### Example 3: Area Class

```dart
class Area {
  double? length;
  double? breadth;

  double calculateArea() {
    return length! * breadth!;
  }
}
```

### Example 4: Student Class

```dart
class Student {
  String? name;
  int? age;
  int? grade;

  void displayInfo() {
    print("Student name: $name.");
    print("Student age: $age.");
    print("Student grade: $grade.");
  }
}
```

## Key Concepts

- Classes serve as blueprints for object creation
- Properties contain object data
- Methods define object behaviors
- The `?` operator enables null safety in Dart
- Creating instances from classes occurs in subsequent lessons

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/class-in-dart/
- **Fetched**: 2026-01-27
