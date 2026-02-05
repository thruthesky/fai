# Dart Constructors: Complete Guide

## Definition

A constructor is "a special method used to initialize an object. It is called automatically when an object is created, and it can be used to set the initial values for the object's properties."

## Key Rules

- Constructor name must match the class name
- Constructors have no return type
- Called automatically upon object creation
- Used to initialize property values

## Constructor Syntax

```dart
class ClassName {
  ClassName() {
    // body of the constructor
  }
}
```

## Types of Constructors

### 1. Basic Constructor

```dart
class Student {
  String? name;
  int? age;

  Student(String name, int age) {
    this.name = name;
    this.age = age;
  }
}
```

### 2. Shorthand Constructor

The concise format assigns parameters directly:

```dart
class Person {
  String? name;
  int? age;

  Person(this.name, this.age);
}
```

### 3. Optional Parameters

Square brackets denote optional parameters with default values:

```dart
class Employee {
  String? name;
  String? subject;

  Employee(this.name, [this.subject = "N/A"]);
}
```

### 4. Named Parameters

Curly braces enable named parameter syntax:

```dart
class Chair {
  String? name;
  String? color;

  Chair({this.name, this.color});
}

// Usage
Chair chair = Chair(name: "Chair1", color: "Red");
```

### 5. Default Values

```dart
class Table {
  String? name;

  Table({this.name = "Table1"});
}
```

## With vs. Without Constructor

**Without constructor:** Manual property assignment required after instantiation

**With constructor:** Properties initialized automatically during object creation

---

**Challenge:** Create a Patient class with name, age, and disease properties, initialize them via constructor, and display the values.

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/constructor-in-dart/
- **Fetched**: 2026-01-27
