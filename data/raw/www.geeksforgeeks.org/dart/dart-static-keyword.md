# Dart - Static Keyword Tutorial

## Overview

The `static` keyword manages memory for global data members in Dart. It applies to class fields and methods, making them belong to the class itself rather than individual instances.

### Key Characteristics

- **Class-level access**: static variables and methods are part of the class instead of a specific instance
- **Single copy sharing**: Only one copy of a static variable exists across all class instances
- **No instantiation needed**: Access directly via the class name without creating an object
- **Persistent values**: Enables data retention between different instances

## Dart Static Variables

Static variables are shared across all class instances. Memory allocation occurs once during class loading.

### Declaration Syntax

```dart
static [data_type] [variable_name];
```

### Access Pattern

```dart
Classname.staticVariable;
```

## Dart Static Methods

Static methods belong to the class rather than instances. They can only access static variables and invoke other static methods—making them ideal for utility functions.

### Declaration Syntax

```dart
static return_type method_name() {
    // Statement(s)
}
```

### Invocation Pattern

```dart
ClassName.staticMethod();
```

## Code Examples

### Static Variable Usage

```dart
class Employee {
    static var emp_dept;
    var emp_name;
    int emp_salary = 0;

    showDetails() {
        print("Name: ${emp_name}");
        print("Salary: ${emp_salary}");
        print("Dept: ${emp_dept}");
    }
}

void main() {
    Employee e1 = new Employee();
    Employee e2 = new Employee();
    Employee.emp_dept = "MIS";

    e1.emp_name = 'Rahul';
    e1.emp_salary = 50000;
    e1.showDetails();

    e2.emp_name = 'Tina';
    e2.emp_salary = 55000;
    e2.showDetails();
}
```

**Output:**
```
Name of the Employee is: Rahul
Salary of the Employee is: 50000
Dept. of the Employee is: MIS
Name of the Employee is: Tina
Salary of the Employee is: 55000
Dept. of the Employee is: MIS
```

### Static Method Usage

```dart
class StaticMem {
    static int num = 0;
    static disp() {
        print("The value of num is ${StaticMem.num}");
    }
}

void main() {
    StaticMem.num = 75;
    StaticMem.disp();
}
```

**Output:**
```
The value of num is 75
```

## Benefits

- Optimized memory usage through single instance sharing
- Simplified access to class-level functionality
- Ideal for constants, utility functions, and shared resources
- Promotes efficient, organized code structure

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-static-keyword/
- **Fetched**: 2026-01-27
