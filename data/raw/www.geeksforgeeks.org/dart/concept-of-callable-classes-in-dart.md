# Callable Classes in Dart

## Overview

Dart enables developers to create **callable classes**, which allow class instances to be invoked as functions. This is accomplished by implementing the `call()` method within the class definition.

## Syntax

```dart
class ClassName {
    // class content

    ReturnType call(Parameters) {
        // call function content
    }
}
```

The `call()` method accepts parameters and returns a value, making instances behave like function objects.

## Basic Example

```dart
class GeeksForGeeks {
    String call(String a, String b, String c)
        => 'Welcome to $a$b$c!';
}

void main() {
    var geek_input = GeeksForGeeks();
    var geek_output = geek_input('Geeks', 'For', 'Geeks');
    print(geek_output); // Output: Welcome to GeeksForGeeks!
}
```

## Practical Implementation

```dart
class Adder {
    int call(int a, int b) {
        return a + b;
    }
}

void main() {
    var adder = Adder();
    var sum = adder(1, 2);
    print(sum); // Output: 3
}
```

## Key Constraints

**Important Limitation**: Dart doesn't support multiple callable methods—attempting to define multiple `call()` methods results in a compilation error about duplicate declarations.

## Alternative: Anonymous Functions

Callable behavior can also be achieved with anonymous functions:

```dart
void main() {
    var adder = (int a, int b) {
        return a + b;
    };

    var sum = adder(1, 2);
    print(sum); // Output: 3
}
```

## Summary

- Callable classes require a single `call()` method implementation
- Instances can then be invoked directly like functions
- Method overloading for `call()` is not permitted
- Anonymous functions provide similar functionality without class definitions

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/concept-of-callable-classes-in-dart/
- **Fetched**: 2026-01-27
