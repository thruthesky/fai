# Dart - Getters and Setters

## Overview

Getters and Setters, also called accessors and mutators, allow the program to initialize and retrieve the values of class fields respectively.

**Key Points:**
- Getters are defined using the `get` keyword
- Setters are defined using the `set` keyword
- Every class has default getter/setter associations
- Custom implementations override defaults

## Syntax

### Getter Definition
```dart
ReturnType get identifier {
    // statements
}
```

### Setter Definition
```dart
set identifier(Type value) {
    // statements
}
```

## Important Characteristics

A getter accepts no parameters and returns a value. In contrast, a setter takes exactly one parameter and returns nothing.

## Example 1: Student Class with Validation

```dart
class Student {
    String name = '';
    int age = 0;

    // Getter for student name
    String get stud_name {
        return name;
    }

    // Setter for student name
    void set stud_name(String name) {
        this.name = name;
    }

    // Setter with validation
    void set stud_age(int age) {
        if (age <= 5) {
            print("Age should be greater than 5");
        } else {
            this.age = age;
        }
    }

    // Getter for student age
    int get stud_age {
        return age;
    }
}

void main() {
    Student s1 = Student();
    s1.stud_name = 'GFG';
    s1.stud_age = 0;
    print(s1.stud_name);  // Output: GFG
    print(s1.stud_age);   // Output: 0
}
```

**Output:**
```
Age should be greater than 5
GFG
0
```

## Example 2: Cat Class with Arrow Syntax

```dart
class Cat {
    bool _isHungry = true;

    // Getter: computed property
    bool get isCuddly => !_isHungry;

    bool get isHungry => _isHungry;

    set isHungry(bool hungry) {
        _isHungry = hungry;
    }
}

void main() {
    var cat = Cat();
    print("Is cat hungry? ${cat.isHungry}");
    print("Is cat cuddly? ${cat.isCuddly}");
    print("Feed cat.");
    cat.isHungry = false;
    print("Is cat hungry? ${cat.isHungry}");
    print("Is cat cuddly? ${cat.isCuddly}");
}
```

**Output:**
```
Is cat hungry? true
Is cat cuddly? false
Feed cat.
Is cat hungry? false
Is cat cuddly? true
```

## Key Takeaways

- Use getters to retrieve computed or stored values
- Use setters to validate and update field values
- Arrow syntax (`=>`) provides concise getter/setter definitions
- Validation logic prevents invalid state assignments
- Private fields (prefixed with `_`) work with getters/setters for encapsulation

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-getters-and-setters/
- **Fetched**: 2026-01-27
