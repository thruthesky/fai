# Constructors in Dart - Complete Tutorial

## Overview

Constructors are a special method that is used to initialize fields when an object is created in Dart. When an instance is instantiated, the constructor automatically executes. Each class has a default constructor provided by the compiler, though you can define custom ones.

## Constructor Syntax

```dart
class_name([parameters]) {
    // Constructor Body
}
```

**Key characteristics:**
- Share the same name as the class
- No return type declaration
- Parameters are optional
- Body executes upon object instantiation
- Replace the default constructor when defined

## Three Constructor Types

### 1. Default Constructor

A parameterless constructor that initializes an object with no arguments.

```dart
class Gfg {
    Gfg() {
        print('This is the default constructor');
    }
}

void main() {
    Gfg geek = new Gfg();
}
```

**Output:** `This is the default constructor`

### 2. Parameterized Constructor

Accepts arguments for custom initialization during instantiation.

```dart
class Gfg {
    Gfg(int a) {
        print('This is the parameterized constructor');
    }
}

void main() {
    Gfg geek = new Gfg(1);
}
```

**Output:** `This is the parameterized constructor`

**Important:** You can't have two constructors with the same name although they have different parameters.

### 3. Named Constructor

Enables multiple constructors with distinct names, solving the single-name limitation.

**Syntax:**
```dart
class_name.constructor_name(parameters) {
    // Body of Constructor
}
```

**Example:**

```dart
class Gfg {
    Gfg() {
        print("This is the default constructor");
    }

    Gfg.constructor1(int a) {
        print('Parameterized constructor with one parameter');
    }

    Gfg.constructor2(int a, int b) {
        print('Parameterized constructor with two parameters');
        print('Value of a + b is ${a + b}');
    }
}

void main() {
    Gfg geek1 = Gfg();
    Gfg geek2 = Gfg.constructor1(1);
    Gfg geek3 = Gfg.constructor2(2, 3);
}
```

**Output:**
```
This is the default constructor
Parameterized constructor with one parameter
Parameterized constructor with two parameters
Value of a + b is 5
```

## Summary

Dart supports three constructor types enabling flexible object initialization. Default constructors require no parameters, parameterized constructors accept arguments, and named constructors provide multiple initialization paths per class, enhancing code organization and clarity.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/constructors-in-dart/
- **Fetched**: 2026-01-27
