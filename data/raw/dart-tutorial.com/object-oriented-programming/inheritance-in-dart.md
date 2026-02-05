# Inheritance in Dart

## Definition

"Inheritance is a sharing of behaviour between two classes." It enables a class to reuse properties and methods from another class using the `extends` keyword.

## Key Concept

Inheritance creates an "is-a" relationship between classes (e.g., Student is a Person, Truck is a Vehicle).

## Basic Syntax

```dart
class ParentClass {
  // Parent class code
}

class ChildClass extends ParentClass {
  // Child class code
}
```

## Terminology

- **Parent Class**: The class being inherited from (also called base class or super class)
- **Child Class**: The class inheriting properties and methods (also called derived class or sub class)

## Types of Inheritance

### 1. Single Inheritance
A class inherits from only one parent class.

```dart
class Car {
  String? name;
  double? price;
}

class Tesla extends Car {
  void display() {
    print("Name: ${name}");
    print("Price: ${price}");
  }
}
```

### 2. Multilevel Inheritance
A class inherits from another class that itself inherits from a third class.

```dart
class Car {
  String? name;
  double? price;
}

class Tesla extends Car {
  void display() {
    print("Name: ${name}");
    print("Price: ${price}");
  }
}

class Model3 extends Tesla {
  String? color;

  void display() {
    super.display();
    print("Color: ${color}");
  }
}
```

**Note**: The `super` keyword calls the parent class method.

### 3. Hierarchical Inheritance
Multiple child classes inherit from a single parent class.

```dart
class Shape {
  double? diameter1;
  double? diameter2;
}

class Rectangle extends Shape {
  double area() {
    return diameter1! * diameter2!;
  }
}

class Triangle extends Shape {
  double area() {
    return 0.5 * diameter1! * diameter2!;
  }
}
```

## Advantages

- "It promotes reusability of the code and reduces redundant code"
- Improves program design and structure
- Reduces maintenance costs and time
- Facilitates creation of class libraries
- Enforces standard interfaces across child classes

## Limitation: No Multiple Inheritance

Dart does not support multiple inheritance because "it can lead to ambiguity." If a class inherited from two parents with identical method names, the system wouldn't know which method to call.

## Why Avoid Copy-Paste Instead?

Duplicating code across classes creates maintenance challenges—changes in one class must be replicated elsewhere, increasing error risk.

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/inheritance-in-dart/
- **Fetched**: 2026-01-27
