# Getter and Setter Methods in Dart

## Overview

**Getter and Setter methods** are class mechanisms for manipulating class field data. The getter retrieves field values, while the setter assigns values to fields in a controlled manner.

## Getter Method

A getter accesses and retrieves a particular class field's data. All classes have default getters, though they can be explicitly overridden.

**Syntax:**
```dart
return_type get field_name {
    ...
}
```

Key points:
- Requires a return type declaration
- No parameters are defined
- Uses the `get` keyword

## Setter Method

A setter enables controlled assignment of values to class fields. All classes possess default setters, but explicit overrides are possible.

**Syntax:**
```dart
set field_name(parameter_type value) {
    ...
}
```

Key points:
- Uses the `set` keyword
- Accepts a single parameter
- Modifies class field data

## Practical Example

```dart
class Gfg {
    // Private field
    String geekName = '';

    // Getter method
    String get getName {
        return geekName;
    }

    // Setter method
    set setName(String name) {
        geekName = name;
    }
}

void main() {
    Gfg geek = Gfg();

    // Using setter
    geek.setName = "GeeksForGeeks";

    // Using getter
    print("Welcome to ${geek.getName}");
}
```

**Output:**
```
Welcome to GeeksForGeeks
```

## Key Benefits

These methods provide:
- **Encapsulation**: Restricts direct field access while enabling controlled operations
- **Data Management**: Ensures modifications follow established constraints
- **Code Quality**: Produces cleaner, more maintainable, and secure implementations

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/getter-and-setter-methods-in-dart/
- **Fetched**: 2026-01-27
