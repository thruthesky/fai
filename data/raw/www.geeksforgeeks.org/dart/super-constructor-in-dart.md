# Super Constructor in Dart - Tutorial

## Overview

In Dart, subclasses inherit properties and methods from parent classes using the `extends` keyword, but they cannot automatically inherit constructors. To initialize the parent class, developers use a **super constructor**. There are two approaches: implicit and explicit invocation.

## Implicit Super Constructor

When a child class is instantiated without explicitly calling the parent constructor, the parent's default constructor is invoked automatically.

### Example: Parent Constructor with No Parameters

```dart
class SuperGeek {
  // Creating parent constructor
  SuperGeek() {
    print("You are inside Parent constructor!!");
  }
}

class SubGeek extends SuperGeek {
  // Creating child constructor
  SubGeek() {
    print("You are inside Child constructor!!");
  }
}

void main() {
  SubGeek geek = new SubGeek();
}
```

**Output:**
```
You are inside Parent constructor!!
You are inside Child constructor!!
```

## Explicit Super Constructor

When the parent constructor requires parameters, you must explicitly invoke it using the `super()` syntax within the child constructor's initializer list.

### Syntax
```dart
Child_class_constructor() : super(parameters) {
  ...
}
```

### Example: Parent Constructor with Parameters

```dart
class SuperGeek {
  // Creating parent constructor with parameter
  SuperGeek(String geek_name) {
    print("You are inside Parent constructor!!");
    print("Welcome to $geek_name");
  }
}

class SubGeek extends SuperGeek {
  // Calling parent class constructor explicitly
  SubGeek() : super("Geeks for Geeks") {
    print("You are inside Child constructor!!");
  }
}

void main() {
  SubGeek geek = new SubGeek();
}
```

**Output:**
```
You are inside Parent constructor!!
Welcome to Geeks for Geeks
You are inside Child constructor!!
```

## Key Takeaways

The `extends` keyword enables inheritance of methods and properties, yet constructors require special handling via the `super` keyword. Implicit invocation occurs automatically with default parent constructors, while explicit calls pass required parameters to parent initialization before executing child constructor logic.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/super-constructor-in-dart/
- **Fetched**: 2026-01-27
