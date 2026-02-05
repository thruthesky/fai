# Static Members in Dart

## Overview

"If you want to define a variable or method that is shared by all instances of a class, you can use the static keyword." Static members enable memory-efficient code by allowing data and behavior to be shared across all class instances rather than duplicated for each object.

## Static Variables

### Declaration

Static variables are declared using the `static` keyword before the variable name:

```dart
class ClassName {
  static dataType variableName;
}
```

### Initialization

Variables are initialized by assigning a value:

```dart
class ClassName {
  static dataType variableName = value;
  // Examples:
  // static int num = 10;
  // static String name = "Dart";
}
```

### Access Pattern

"You need to use the ClassName.variableName to access a static variable in Dart."

```dart
class ClassName {
  static dataType variableName = value;

  void display() {
    print(variableName);
  }
}

void main() {
  dataType value = ClassName.variableName;
}
```

## Static Variable Examples

### Example 1: Employee Counter

This demonstrates how static variables track class-level data across instances:

```dart
class Employee {
  static int count = 0;

  Employee() {
    count++;
  }

  void totalEmployee() {
    print("Total Employee: $count");
  }
}

void main() {
  Employee e1 = new Employee();
  e1.totalEmployee(); // Total Employee: 1
  Employee e2 = new Employee();
  e2.totalEmployee(); // Total Employee: 2
  Employee e3 = new Employee();
  e3.totalEmployee(); // Total Employee: 3
}
```

### Example 2: Shared School Information

Static variables efficiently store data common to all instances:

```dart
class Student {
  int id;
  String name;
  static String schoolName = "ABC School";

  Student(this.id, this.name);

  void display() {
    print("Id: ${this.id}");
    print("Name: ${this.name}");
    print("School Name: ${Student.schoolName}");
  }
}

void main() {
  Student s1 = new Student(1, "John");
  s1.display();
  Student s2 = new Student(2, "Smith");
  s2.display();
}
```

## Static Methods

### Syntax

"A static method is shared by all instances of a class... You can access a static method without creating an object of the class."

```dart
class ClassName {
  static returnType methodName() {
    //statements
  }
}
```

### Example 3: Interest Calculator

```dart
class SimpleInterest {
  static double calculateInterest(double principal, double rate, double time) {
    return (principal * rate * time) / 100;
  }
}

void main() {
  print("The simple interest is ${SimpleInterest.calculateInterest(1000, 2, 2)}");
  // Output: The simple interest is 40.0
}
```

### Example 4: Password Generator

```dart
import 'dart:math';

class PasswordGenerator {
  static String generateRandomPassword() {
    List<String> allalphabets = 'abcdefghijklmnopqrstuvwxyz'.split('');
    List<int> numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    List<String> specialCharacters = ["@", "#", "%", "&", "*"];
    List<String> password = [];

    for (int i = 0; i < 5; i++) {
      password.add(allalphabets[Random().nextInt(allalphabets.length)]);
      password.add(numbers[Random().nextInt(numbers.length)].toString());
      password.add(specialCharacters[Random().nextInt(specialCharacters.length)]);
    }
    return password.join();
  }
}

void main() {
  print(PasswordGenerator.generateRandomPassword());
}
```

## Key Takeaways

- "Static members are accessed using the class name."
- "All instances of a class share static members."
- No object instantiation is required to access static members
- Static variables initialize only once when the class loads
- Ideal for class-level data and utility functions

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/static-in-dart/
- **Fetched**: 2026-01-27
