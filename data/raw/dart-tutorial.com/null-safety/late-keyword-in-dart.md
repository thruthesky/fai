# Late Keyword in Dart

## Overview

The `late` keyword in Dart declares a non-nullable variable or field that will be initialized at a later time, rather than at declaration. According to the tutorial, when you use `late`, you're telling Dart: "Don't assign that variable a value yet. You will assign value later. You will make sure the variable has a value before you use it."

## Basic Examples

### Example 1: Simple Late Variable

```dart
late String name;

void main() {
  name = "John";
  print(name);
}
```
Output: `John`

### Example 2: Late Variable in a Class

```dart
class Person {
  late String name;

  void greet() {
    print("Hello $name");
  }
}

void main() {
  Person person = Person();
  person.name = "John";
  person.greet();
}
```
Output: `Hello John`

## Use Cases

The tutorial identifies two primary use cases:

1. **Non-nullable variable declaration** - Declaring variables that aren't initialized immediately but are guaranteed before use
2. **Lazy initialization** - Delaying expensive computations or object creation until actually needed

## Lazy Initialization

Lazy initialization is "a design pattern that delays the creation of an object, the calculation of a value, or some other expensive process until the first time you need it."

### Example 3: Lazy Initialization with Functions

```dart
String provideCountry() {
  print("Function is called");
  return "USA";
}

void main() {
  print("Starting");
  late String value = provideCountry();
  print("End");
  print(value);
}
```

Without `late`, the function executes immediately. With `late`, output is:
```
Starting
End
Function is called
USA
```

### Example 4: Late Variables in Class Initialization

```dart
class Person {
  final int age;
  final String name;
  late String description = heavyComputation();

  Person(this.age, this.name) {
    print("Constructor is called");
  }

  String heavyComputation() {
    print("heavyComputation is called");
    return "Heavy Computation";
  }
}

void main() {
  Person person = Person(10, "John");
  print(person.name);
  print(person.description);
}
```

Output:
```
Constructor is called
John
heavyComputation is called
Heavy Computation
```

### Example 5: Dependent Late Variables

```dart
class Person {
  late String fullName = _getFullName();
  late String firstName = fullName.split(" ").first;
  late String lastName = fullName.split(" ").last;

  String _getFullName() {
    print("_getFullName is called");
    return "John Doe";
  }
}

void main() {
  print("Start");
  Person person = Person();
  print("First Name: ${person.firstName}");
  print("Last Name: ${person.lastName}");
  print("Full Name: ${person.fullName}");
  print("End");
}
```

Output:
```
Start
_getFullName is called
First Name: John
Last Name: Doe
Full Name: John Doe
End
```

## Late Final Keyword

The `late final` combination restricts a variable to a single assignment after declaration.

### Example 6: Late Final Variable

```dart
class Student {
  late final String name;

  Student(this.name);
}

void main() {
  Student student = Student("John");
  print(student.name);
  student.name = "Doe"; // Error
}
```

Output:
```
John
Unhandled exception:
LateInitializationError: Field 'name' has already been initialized.
```

## Important Note

The tutorial emphasizes that using `late` is "a contract between you and Dart." If you don't assign a value before using the variable, Dart will throw a `LateInitializationError`.

---

## Source

- **URL**: https://dart-tutorial.com/null-safety/late-keyword-in-dart/
- **Fetched**: 2026-01-27
