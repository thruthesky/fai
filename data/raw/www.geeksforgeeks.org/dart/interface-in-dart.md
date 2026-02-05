# Interface in Dart

## Overview

An interface in Dart provides a blueprint for classes to follow. The interface in the dart provides the user with the blueprint of the class, which any class should follow if it interfaces that class.

Dart doesn't have a direct interface keyword. Instead, classes serve as interfaces, and you use the `implements` keyword to enforce the contract.

## Syntax

```dart
class InterfaceClassName {
   ...
}

class ClassName implements InterfaceClassName {
   ...
}
```

## Basic Example

```dart
// Interface class
class Geek {
    void printdata() {
        print("Hello Geek !!");
    }
}

// Implementation class
class Gfg implements Geek {
    void printdata() {
        print("Welcome to GeeksForGeeks");
    }
}

void main() {
    Gfg geek1 = new Gfg();
    geek1.printdata();
}
```

**Output:**
```
Welcome to GeeksForGeeks
```

## Multiple Interfaces

Dart supports multiple interface implementation through the `implements` keyword. A class can implement multiple interfaces simultaneously:

```dart
class Interface1 {
    void method1() { print("Interface1"); }
}

class Interface2 {
    void method2() { print("Interface2"); }
}

class Implementation implements Interface1, Interface2 {
    void method1() { print("Method 1 implemented"); }
    void method2() { print("Method 2 implemented"); }
}
```

## Key Points

- **No direct interface syntax**: Classes are used as interfaces in Dart
- **Method overriding required**: All of its method and instance variable must be overridden during the interface
- **Flexible inheritance**: A class can extend only one class but implement multiple interfaces
- **Achieves abstraction**: Interfaces provide a contract that implementing classes must follow

## Importance

- Enables abstraction mechanisms
- Facilitates multiple inheritance patterns
- Establishes clear contracts between classes

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/interface-in-dart/
- **Fetched**: 2026-01-27
