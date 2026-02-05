# Inheritance Of Constructor in Dart

## Overview

"Inheritance of constructor in Dart is a process of inheriting the constructor of the parent class to the child class." This allows code reuse from parent classes.

## Key Concept

When a child class is instantiated, the parent class constructor executes first, followed by the child class constructor.

## Example 1: Basic Constructor Inheritance

```dart
class Laptop {
  Laptop() {
    print("Laptop constructor");
  }
}

class MacBook extends Laptop {
  MacBook() {
    print("MacBook constructor");
  }
}

void main() {
  var macbook = MacBook();
}
```

**Output:**
```
Laptop constructor
MacBook constructor
```

## Example 2: Constructors With Parameters

When passing parameters to a parent constructor, use the `super` keyword:

```dart
class Laptop {
  Laptop(String name, String color) {
    print("Laptop constructor");
    print("Name: $name");
    print("Color: $color");
  }
}

class MacBook extends Laptop {
  MacBook(String name, String color) : super(name, color) {
    print("MacBook constructor");
  }
}

void main() {
  var macbook = MacBook("MacBook Pro", "Silver");
}
```

## Example 3: Inheritance With Additional Properties

```dart
class Person {
  String name;
  int age;
  Person(this.name, this.age);
}

class Student extends Person {
  int rollNumber;
  Student(String name, int age, this.rollNumber) : super(name, age);
}

void main() {
  var student = Student("John", 20, 1);
  print("Student name: ${student.name}");
  print("Student age: ${student.age}");
  print("Student roll number: ${student.rollNumber}");
}
```

## Example 4: Named Parameters

```dart
class Laptop {
  Laptop({String name, String color}) {
    print("Laptop constructor");
    print("Name: $name");
    print("Color: $color");
  }
}

class MacBook extends Laptop {
  MacBook({String name, String color}) : super(name: name, color: color) {
    print("MacBook constructor");
  }
}

void main() {
  var macbook = MacBook(name: "MacBook Pro", color: "Silver");
}
```

## Example 5: Calling Named Parent Constructors

You can invoke named constructors from parent classes using `super`:

```dart
class Laptop {
  Laptop() {
    print("Laptop constructor");
  }

  Laptop.named() {
    print("Laptop named constructor");
  }
}

class MacBook extends Laptop {
  MacBook() : super.named() {
    print("MacBook constructor");
  }
}

void main() {
  var macbook = MacBook();
}
```

**Output:**
```
Laptop named constructor
MacBook constructor
```

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/inheritance-of-constructor-in-dart/
- **Fetched**: 2026-01-27
