# Parameterized Constructor in Dart

## Overview

"Parameterized constructor is used to initialize the instance variables of the class." This constructor type accepts parameters that are passed during object creation to set initial values for class properties.

## Syntax

The basic structure follows this pattern:

```dart
class ClassName {
  // Instance Variables
  int? number;
  String? name;
  // Parameterized Constructor
  ClassName(this.number, this.name);
}
```

## Example 1: Basic Parameterized Constructor

A class with multiple properties initialized through the constructor:

```dart
class Student {
  String? name;
  int? age;
  int? rollNumber;
  // Constructor
  Student(this.name, this.age, this.rollNumber);
}

void main(){
    Student student = Student("John", 20, 1);
    print("Name: ${student.name}");
    print("Age: ${student.age}");
    print("Roll Number: ${student.rollNumber}");
}
```

**Output:**
```
Name: John
Age: 20
Roll Number: 1
```

## Example 2: Named Parameters Approach

Using named parameters provides flexibility in argument passing order:

```dart
class Student {
  String? name;
  int? age;
  int? rollNumber;

  Student({String? name, int? age, int? rollNumber}) {
    this.name = name;
    this.age = age;
    this.rollNumber = rollNumber;
  }
}

void main(){
    Student student = Student(name: "John", age: 20, rollNumber: 1);
    print("Name: ${student.name}");
    print("Age: ${student.age}");
    print("Roll Number: ${student.rollNumber}");
}
```

**Output:**
```
Name: John
Age: 20
Roll Number: 1
```

## Example 3: Default Values

Constructor parameters can include default values:

```dart
class Student {
  String? name;
  int? age;

  Student({String? name = "John", int? age = 0}) {
    this.name = name;
    this.age = age;
  }
}

void main(){
    Student student = Student();
    print("Name: ${student.name}");
    print("Age: ${student.age}");
}
```

**Output:**
```
Name: John
Age: 0
```

## Key Note

"At the time of object creation, you must pass the parameters through the constructor which initialize the variables value, avoiding the null values."

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/parameterized-constructor-in-dart/
- **Fetched**: 2026-01-27
