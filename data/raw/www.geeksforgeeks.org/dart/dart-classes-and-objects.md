# Dart - Classes And Objects Tutorial

## Overview

Dart is an object-oriented programming language supporting classes and objects. Developers define custom classes using the `class` keyword to organize code effectively.

## Classes in Dart

### Definition

Class is the blueprint of objects, and class is the collection of data members and data function means, which include these fields, getter and setter, and constructor and functions.

### Syntax

```dart
class class_name {
  // Body of class
}
```

The `class` keyword initiates the class definition, `class_name` provides the identifier, and the body contains fields, constructors, getter/setter methods, and other components.

### Three Core Components

1. **Class Fields** - Variables holding object data
2. **Class Methods** - Functions providing object behavior
3. **Constructors** - Code blocks initializing object state during creation

## Class Fields Example

```dart
class Student {
  // Properties of Class
  int? roll_no;
  String? name;
}
```

Fields store data specific to each object instance.

## Class Methods Example

```dart
class Student {
  int? roll_no;
  String? name;

  void print_name(){
    print("Student Name: $name");
  }
}
```

Methods define behaviors and operations available on the class.

## Constructors

A Constructor is a block of code that initializes the state and values during object creation. Constructor is name same as the class name and doesn't return any value.

```dart
class_name( [ parameters ] ) {
  // Constructor Body
}
```

## Objects in Dart

Objects represent instances of classes. Declaration uses the `new` keyword (optional in Dart 2+):

```dart
var object_name = new class_name([ arguments ]);
// or
var objectName = ClassName([ arguments ]);
```

**Syntax Components:**
- `new` - Creates class instances
- `object_name` - Instance identifier (follows variable naming conventions)
- `class_name` - Target class identifier
- `arguments` - Constructor parameters (if required)

### Accessing Properties and Methods

```dart
// For accessing the property
object_name.property_name;

// For accessing the method
object_name.method_name();
```

Use the dot operator (`.`) to access class members.

## Practical Example

```dart
// Creating Class named Gfg
class Gfg {
  // Creating Field inside the class
  String geek1 = '';

  // Creating Function inside class
  void geek() {
    print("Welcome to $geek1");
  }
}

void main() {
  // Creating Instance of class
  Gfg geek = new Gfg();

  // Assigning value to geek1
  geek.geek1 = 'GeeksforGeeks';

  // Calling function
  geek.geek();
}
```

**Output:**
```
Welcome to GeeksforGeeks
```

## Key Takeaways

Dart's object-oriented programming (OOP) approach facilitates efficient code organization and promotes reusability. Understanding classes, objects, constructors, fields, and methods is essential for building scalable applications.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-classes-and-objects/
- **Fetched**: 2026-01-27
