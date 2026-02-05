# Dart - this Keyword Tutorial

## Overview

The `this` keyword in Dart serves as an implicit object pointing to the current class object. It refers to the present instance within methods or constructors, primarily resolving naming conflicts between class attributes and parameters.

## Primary Uses

The `this` keyword enables several essential functions:

- Referencing instance variables of the current class
- Initiating current class constructors
- Passing as method arguments
- Passing as constructor arguments
- Invoking current class methods
- Returning the current class instance

## Code Example 1: Referring to Instance Variables

```dart
void main() {
    Student s1 = new Student('S001');
}

class Student {
    var st_id;
    Student(var st_id) {
        this.st_id = st_id;
        print("GFG - Dart THIS Example");
        print("The Student ID is : ${st_id}");
    }
}
```

**Output:**
```
GFG - Dart THIS Example
The Student ID is : S001
```

## Code Example 2: Current Class Instance

```dart
void main() {
    mob m1 = new mob();
    m1.Car("M101");
}

class mob {
    String mobile = "";
    Car(String mobile) {
        this.mobile = mobile;
        print("The mobile is : ${mobile}");
    }
}
```

**Output:**
```
The mobile is : M101
```

## Key Takeaway

Utilizing `this` enhances code clarity and eliminates ambiguity. The keyword facilitates cleaner, more maintainable Dart applications by allowing developers to explicitly reference current instance members while enabling constructor chaining and method invocation within identical instances.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-this-keyword/
- **Fetched**: 2026-01-27
