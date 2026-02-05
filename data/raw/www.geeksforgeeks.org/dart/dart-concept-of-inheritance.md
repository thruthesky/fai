# Dart - Concept of Inheritance

## Introduction

In Dart, classes can inherit from other classes, enabling code reuse through the `extend` keyword. This creates a hierarchical relationship between parent and child classes.

## Terminology

- **Parent Class (Base/Super Class)**: The class whose properties are inherited by child classes
- **Child Class (Derived/Sub Class)**: The class that inherits properties from parent classes

## Basic Syntax

```dart
// Parent class
class ParentClass {
  // properties and methods
}

// Child class
class ChildClass extends ParentClass {
  // inherits from ParentClass
}
```

## Types of Inheritance

### 1. Single Inheritance

A subclass inherits from one superclass. The child class gains all properties and behaviors of its parent.

**Example:**

```dart
// Parent class
class Gfg {
  void output() {
    print("Welcome to gfg!!\nYou are inside output function.");
  }
}

// Child class
class GfgChild extends Gfg {
  // inherits output() method
}

void main() {
  var geek = new GfgChild();
  geek.output(); // calls inherited method
}
```

**Output:**
```
Welcome to gfg!!
You are inside output function.
```

### 2. Multilevel Inheritance

A derived class inherits from a base class, which itself acts as a base for other classes, creating inheritance chains.

**Example:**

```dart
class Gfg {
  void output1() {
    print("Welcome to gfg!!\nYou are inside the output function of Gfg class.");
  }
}

class GfgChild1 extends Gfg {
  void output2() {
    print("Welcome to gfg!!\nYou are inside the output function of GfgChild1 class.");
  }
}

class GfgChild2 extends GfgChild1 {
  // inherits from GfgChild1, which inherits from Gfg
}

void main() {
  var geek = new GfgChild2();
  geek.output1(); // from Gfg
  geek.output2(); // from GfgChild1
}
```

### 3. Hierarchical Inheritance

Multiple child classes inherit from a single parent class.

**Example:**

```dart
class Gfg {
  void output1() {
    print("Welcome to gfg!!\nYou are inside output function of Gfg class.");
  }
}

class GfgChild1 extends Gfg {}
class GfgChild2 extends Gfg {}
class GfgChild3 extends Gfg {}

void main() {
  var geek1 = new GfgChild1();
  geek1.output1();

  var geek2 = new GfgChild2();
  geek2.output1();

  var geek3 = new GfgChild3();
  geek3.output1();
}
```

### 4. Multiple Inheritance (Not Supported)

Dart does not support multiple inheritance directly but provides mixins as an alternative solution for reusing code from multiple sources.

**Mixin Alternative:**

```dart
mixin A {
  void showA() {
    print("This is class A.");
  }
}

mixin B {
  void showB() {
    print("This is class B.");
  }
}

class C with A, B {
  void showC() {
    print("This is class C.");
  }
}

void main() {
  C obj = C();
  obj.showA();
  obj.showB();
  obj.showC();
}
```

**Output:**
```
This is class A.
This is class B.
This is class C.
```

## Important Points

- Child classes inherit all properties and methods **except constructors** from parent classes
- Like Java, Dart also doesn't support multiple inheritance
- Use `extends` for single inheritance chains
- Use `with` for mixins to achieve multiple inheritance functionality

## Conclusion

Inheritance enables code reuse and structured class hierarchies through the `extends` keyword. Dart supports single, multilevel, and hierarchical inheritance patterns while restricting multiple inheritance to maintain simplicity. Mixins provide an alternative mechanism for sharing functionality across multiple classes.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-concept-of-inheritance/
- **Fetched**: 2026-01-27
