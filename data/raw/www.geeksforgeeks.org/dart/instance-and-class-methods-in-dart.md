# Instance and Class Methods in Dart

## Overview

Dart enables developers to create custom methods that perform specific actions within classes. These methods enhance code organization and reusability while reducing program complexity. Methods may or may not return values and can accept zero or more parameters.

## Two Types of Methods

Dart supports two primary method categories:

1. **Instance Methods**
2. **Class Methods**

---

## Instance Methods

### Definition

Instance methods are methods not declared with the `static` keyword. They can access instance variables and require object instantiation before invocation.

### Syntax

```dart
// Declaring instance method
return_type method_name() {
  // Body of method
}

// Creating object
class_name object_name = new class_name();

// Calling instance method
object_name.method_name();
```

### Example

```dart
class Gfg {
  // Instance variables
  int a = 0;
  int b = 0;

  // Instance method
  void sum(int c, int d) {
    this.a = c;
    this.b = d;
    print('Sum of numbers is ${a + b}');
  }
}

void main() {
  // Create object
  Gfg geek = new Gfg();

  // Call method using object
  geek.sum(21, 12);
}
```

**Output:**
```
Sum of numbers is 33
```

---

## Class Methods

### Definition

Class methods are declared with the `static` keyword. They cannot access non-static variables or invoke non-static methods. Unlike instance methods, they can be called directly using the class name without object creation.

### Syntax

```dart
// Creating class method
static return_type method_name() {
  // Body of method
}

// Calling class method
class_name.method_name();
```

### Example

```dart
class Gfg {
  // Class method
  static void sum(int c, int d) {
    print('Sum of numbers is ${c + d}');
  }
}

void main() {
  // Call method using class name
  Gfg.sum(11, 32);
}
```

**Output:**
```
Sum of numbers is 43
```

---

## Key Differences

| Aspect | Instance Methods | Class Methods |
|--------|-----------------|---------------|
| Declaration | No `static` keyword | Uses `static` keyword |
| Access to Instance Variables | Yes | No |
| Object Required | Yes | No |
| Invocation | `object.method()` | `ClassName.method()` |

---

## Conclusion

Understanding the distinction between these method types is crucial for efficient Dart development. Instance methods work with object state through instance variables, while class methods provide shared functionality accessible via the class itself.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/instance-and-class-methods-in-dart/
- **Fetched**: 2026-01-27
