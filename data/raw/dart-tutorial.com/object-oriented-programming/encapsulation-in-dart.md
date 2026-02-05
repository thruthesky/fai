# Encapsulation in Dart

## Overview

"Encapsulation means hiding data within a library, preventing it from outside factors." This fundamental OOP concept helps manage program complexity and control data access.

## Key Concepts

### What is a Library?

By default, every `.dart` file functions as a library—a collection of functions and classes that can be imported into other libraries using the `import` keyword.

### How to Achieve Encapsulation

Two main strategies:

1. Declare class properties as private using underscore (`_`)
2. Provide public getter and setter methods to access/update private properties

**Important Note:** "Dart doesn't support keywords like public, private, and protected. Dart uses _ (underscore) to make a property or method private."

## Code Examples

### Example 1: Basic Encapsulation

```dart
class Employee {
  int? _id;
  String? _name;

  int getId() {
    return _id!;
  }

  String getName() {
    return _name!;
  }

  void setId(int id) {
    this._id = id;
  }

  void setName(String name) {
    this._name = name;
  }
}

void main() {
  Employee emp = new Employee();
  emp.setId(1);
  emp.setName("John");

  print("Id: ${emp.getId()}");
  print("Name: ${emp.getName()}");
}
```

**Output:**
```
Id: 1
Name: John
```

### Example 2: Private Properties

```dart
class Employee {
  var _name;

  String getName() {
    return _name;
  }

  void setName(String name) {
    this._name = name;
  }
}

void main() {
  var employee = Employee();
  employee.setName("Jack");
  print(employee.getName());
}
```

**Output:**
```
Jack
```

## Library-Level Privacy

A critical distinction: underscore prefixes create **library-private** properties, not class-private ones. This means properties remain accessible within the same file but are hidden from external libraries.

To enforce true privacy, create separate files for classes and import them accordingly.

## Read-Only Properties

Using the `final` keyword restricts properties to read-only access:

```dart
class Student {
  final _schoolname = "ABC School";

  String getSchoolName() {
    return _schoolname;
  }
}

void main() {
  var student = Student();
  print(student.getSchoolName());
  // student._schoolname = "XYZ School"; // Not allowed
}
```

**Output:**
```
ABC School
```

## Modern Getter and Setter Methods

Use `get` and `set` keywords for cleaner syntax:

```dart
class Vehicle {
  String _model;
  int _year;

  String get model => _model;
  set model(String model) => _model = model;

  int get year => _year;
  set year(int year) => _year = year;
}

void main() {
  var vehicle = Vehicle();
  vehicle.model = "Toyota";
  vehicle.year = 2019;
  print(vehicle.model);
  print(vehicle.year);
}
```

**Output:**
```
Toyota
2019
```

## Why Encapsulation Matters

- **Data Hiding:** Prevents unauthorized external access to internal class data
- **Testability:** Enables isolated class testing without external dependencies
- **Flexibility:** Implementation changes don't affect external code
- **Security:** Restricts member access from outside the library scope

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/encapsulation-in-dart/
- **Fetched**: 2026-01-27
