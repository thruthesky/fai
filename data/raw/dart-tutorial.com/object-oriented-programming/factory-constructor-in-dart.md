# Factory Constructors in Dart

## Introduction

Factory constructors are a specialized constructor type in Dart that provide enhanced flexibility for object creation compared to generative constructors.

## Definition

A factory constructor differs from standard constructors by allowing developers to "return an instance of the class or even subclass" and "return the cached instance of the class."

## Syntax

```dart
class ClassName {
  factory ClassName() {
    // TODO: return ClassName instance
  }

  factory ClassName.namedConstructor() {
    // TODO: return ClassName instance
  }
}
```

## Rules for Factory Constructors

- Must return an instance of the class or sub-class
- Cannot use the `this` keyword
- Can be named or unnamed, invoked like normal constructors
- Cannot access instance members of the class

## Key Use Cases

### 1. Input Validation

Factory constructors enable validation logic that cannot be handled in initializer lists:

```dart
class Area {
  final int length;
  final int breadth;
  final int area;

  const Area._internal(this.length, this.breadth) : area = length * breadth;

  factory Area(int length, int breadth) {
    if (length < 0 || breadth < 0) {
      throw Exception("Length and breadth must be positive");
    }
    return Area._internal(length, breadth);
  }
}
```

### 2. Creating from Maps

Factory constructors facilitate object construction from alternative data sources:

```dart
class Person {
  String firstName;
  String lastName;

  Person(this.firstName, this.lastName);

  factory Person.fromMap(Map<String, Object> map) {
    final firstName = map['firstName'] as String;
    final lastName = map['lastName'] as String;
    return Person(firstName, lastName);
  }
}
```

### 3. Polymorphic Object Creation

Factory constructors can return appropriate subclass instances based on parameters:

```dart
enum ShapeType { circle, rectangle }

abstract class Shape {
  factory Shape(ShapeType type) {
    switch (type) {
      case ShapeType.circle:
        return Circle();
      case ShapeType.rectangle:
        return Rectangle();
      default:
        throw 'Invalid shape type';
    }
  }
  void draw();
}

class Circle implements Shape {
  @override
  void draw() {
    print('Drawing circle');
  }
}
```

### 4. Caching Implementation

Factory constructors enable instance caching to manage object creation:

```dart
class Person {
  final String name;

  Person._internal(this.name);

  static final Map<String, Person> _cache = <String, Person>{};

  factory Person(String name) {
    if (_cache.containsKey(name)) {
      return _cache[name]!;
    } else {
      final person = Person._internal(name);
      _cache[name] = person;
      return person;
    }
  }
}
```

## Singleton Pattern

Factory constructors are ideal for implementing singletons, allowing "only one instance and provides a global point of access to it":

```dart
class Singleton {
  static final Singleton _instance = Singleton._internal();

  factory Singleton() {
    return _instance;
  }

  Singleton._internal();
}

void main() {
  Singleton obj1 = Singleton();
  Singleton obj2 = Singleton();
  print(obj1.hashCode); // Same hash code
  print(obj2.hashCode); // Same hash code
}
```

## Key Advantages

- Implement factory design patterns with conditional subclass instantiation
- Create singleton patterns for shared resource management
- Initialize final variables using complex logic
- Validate inputs before instance creation
- Support alternative construction methods

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/factory-constructor-in-dart/
- **Fetched**: 2026-01-27
