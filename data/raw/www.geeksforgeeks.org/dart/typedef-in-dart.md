# Typedef in Dart

## Overview

Typedef creates user-defined aliases for functions in Dart, enabling cleaner code and better type management. It is used to create a user-defined identity (alias) for a function.

## Why Use Typedef

The primary benefits include:

- **Readability**: Simplifies complex function types for easier understanding
- **Reusability**: Facilitates the definition of versatile function signatures
- **Type Safety**: Allows for clean and safe function parameterization
- **Callbacks**: Perfect for working with callbacks, event handlers, and functional programming patterns

## Basic Syntax

### 1. Defining a Typedef
```dart
typedef FunctionName(int a, int b);
```

### 2. Assigning to a Variable
```dart
typedef variableName = functionName;
```

### 3. Invoking the Typedef
```dart
variableName(parameters);
```

## Code Examples

### Example 1: Basic Typedef Usage

```dart
typedef GeeksForGeeks(int a, int b);

Geek1(int a, int b) {
    print("This is Geek1");
    print("$a and $b are lucky geek numbers !!");
}

Geek2(int a, int b) {
    print("This is Geek2");
    print("$a + $b is equal to ${a + b}.");
}

void main() {
    GeeksForGeeks number = Geek1;
    number(1, 2);

    number = Geek2;
    number(3, 4);
}
```

**Output:**
```
This is Geek1
1 and 2 are lucky geek numbers !!
This is Geek2
3 + 4 is equal to 7.
```

### Example 2: Typedef as Function Parameter

```dart
typedef GeeksForGeeks(int a, int b);

Geek1(int a, int b) {
    print("This is Geek1");
    print("$a and $b are lucky geek numbers !!");
}

number(int a, int b, GeeksForGeeks geek) {
    print("Welcome to GeeksForGeeks");
    geek(a, b);
}

void main() {
    number(21, 23, Geek1);
}
```

**Output:**
```
Welcome to GeeksForGeeks
This is Geek1
21 and 23 are lucky geek numbers !!
```

## Typedef for Collections

Typedef extends beyond functions to variables and collections:

```dart
typedef ListInteger = List<int>;

void main() {
    ListInteger x = [11, 21, 31];
    print(x);
}
```

**Output:**
```
[11, 21, 31]
```

## Important Notes

- Typedefs were restricted to function types before Dart version 2.13
- Modern Dart allows typedef for any type alias, not just functions
- Particularly useful for callback patterns and functional programming approaches

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/typedef-in-dart/
- **Fetched**: 2026-01-27
