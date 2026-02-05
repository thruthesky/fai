# Method Overriding in Dart

## Definition

Method overriding occurs in Dart when a child class tries to override the parent class's method. When a child class extends a parent class, it gains access to inherited methods and can redefine them with new implementations.

## Key Concept

This technique enables different behavior implementations across child classes while maintaining the same method signature, enhancing code flexibility and reusability.

## Important Rules

1. **Child Class Only**: Methods can only be overridden in child classes, not parent classes
2. **Signature Match**: Both parent and child methods must have identical names and parameter lists
3. **Final Methods**: Methods declared as `final` or `static` cannot be overridden
4. **Constructor Restriction**: Parent class constructors cannot be inherited or overridden

## Basic Example

```dart
class SuperGeek {
  void show() {
    print("This is class SuperGeek.");
  }
}

class SubGeek extends SuperGeek {
  void show() {
    print("This is class SubGeek child of SuperGeek.");
  }
}

void main() {
  SuperGeek geek1 = new SuperGeek();
  SubGeek geek2 = new SubGeek();

  geek1.show();  // Output: This is class SuperGeek.
  geek2.show();  // Output: This is class SubGeek child of SuperGeek.
}
```

## Multiple Child Classes Example

```dart
class SuperGeek {
  void show() {
    print("This is class SuperGeek.");
  }
}

class SubGeek1 extends SuperGeek {
  void show() {
    print("This is class SubGeek1 child of SuperGeek.");
  }
}

class SubGeek2 extends SuperGeek {
  void show() {
    print("This is class SubGeek2 child of SuperGeek.");
  }
}

void main() {
  SuperGeek geek1 = new SuperGeek();
  SubGeek1 geek2 = new SubGeek1();
  SubGeek2 geek3 = new SubGeek2();

  geek1.show();
  geek2.show();
  geek3.show();
}
```

**Output:**
```
This is class SuperGeek.
This is class SubGeek1 child of SuperGeek.
This is class SubGeek2 child of SuperGeek.
```

## Benefits

Method overriding supports modular, maintainable object-oriented programming by allowing specialized implementations in child classes while preserving inherited functionality.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/method-overriding-in-dart/
- **Fetched**: 2026-01-27
