# Default Constructor in Dart

## Definition

A default constructor is "automatically created by the dart compiler if you don't create a constructor" and "has no parameters." It is declared using the class name followed by parentheses `()`.

## Key Characteristics

- Automatically invoked when creating a class object
- Used to initialize instance variables
- Takes no parameters
- Called automatically without explicit invocation

## Example 1: Basic Default Constructor

```dart
class Laptop {
  String? brand;
  int? price;

  // Constructor
  Laptop() {
    print("This is a default constructor");
  }
}

void main() {
  Laptop laptop = Laptop();
}
```

**Output:**
```
This is a default constructor
```

## Example 2: Initializing Properties

```dart
class Student {
  String? name;
  int? age;
  String? schoolname;
  String? grade;

  // Default Constructor
  Student() {
    print("Constructor called");
    schoolname = "ABC School";
  }
}

void main() {
  Student student = Student();
  student.name = "John";
  student.age = 10;
  student.grade = "A";
  print("Name: ${student.name}");
  print("Age: ${student.age}");
  print("School Name: ${student.schoolname}");
  print("Grade: ${student.grade}");
}
```

**Output:**
```
Constructor called
Name: John
Age: 10
School Name: ABC School
Grade: A
```

## Practical Challenge

Create a `Person` class with `name` and `planet` properties. Use a default constructor to set `planet` to "earth", then instantiate an object and display both properties.

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/default-constructor-in-dart/
- **Fetched**: 2026-01-27
