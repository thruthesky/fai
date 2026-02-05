# Anonymous Function in Dart

## Overview

This tutorial covers anonymous functions in Dart—functions that lack a name, distinguished from named functions like `main()` or `add()`.

## What is an Anonymous Function?

An anonymous function is created by removing the return type and function name from a standard function declaration. As the tutorial states, "If you remove the return type and the function name, the function is called anonymous function."

## Syntax

The basic structure follows this pattern:

```dart
(parameterList){
  // statements
}
```

## Example 1: Using forEach with Anonymous Functions

This demonstrates iterating through a list using an unnamed function:

```dart
void main() {
  const fruits = ["Apple", "Mango", "Banana", "Orange"];

  fruits.forEach((fruit) {
    print(fruit);
  });
}
```

**Output:**
```
Apple
Mango
Banana
Orange
```

## Example 2: Anonymous Function Assigned to a Variable

This shows storing an anonymous function in a variable for reuse:

```dart
void main() {
  var cube = (int number) {
    return number * number * number;
  };

  print("The cube of 2 is ${cube(2)}");
  print("The cube of 3 is ${cube(3)}");
}
```

**Output:**
```
The cube of 2 is 8
The cube of 3 is 27
```

## Key Takeaways

Anonymous functions are useful for inline operations, particularly with collection methods like `forEach`, and can be assigned to variables for repeated invocation without requiring a formal function declaration.

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/anonymous-function-in-dart/
- **Fetched**: 2026-01-27
