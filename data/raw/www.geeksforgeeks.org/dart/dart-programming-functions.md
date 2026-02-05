# Dart Functions Tutorial

## Overview

The function is a set of statements that take inputs, do some specific computation and produces output. Functions enable code reuse and help organize complex programs into manageable components.

## Defining Functions in Dart

### Basic Syntax

Functions in Dart require:
- **function_name**: The identifier for the function
- **return_type**: The datatype of the output
- **return value**: What the function produces

### Calling Functions

Functions are invoked by their name followed by parentheses containing any required arguments.

## Code Examples

### Example 1: Simple Addition Function

```dart
int add(int a, int b) {
    // Creating function
    int result = a + b;

    // returning value result
    return result;
}

void main() {
    // Calling the function
    var output = add(10, 20);

    // Printing output
    print(output);
}
```

**Output:** `30`

> Note: Function names must be unique; two functions cannot share identical names.

### Example 2: Function Without Parameters or Return Value

```dart
void GFG() {
    // Creating function
    print("Welcome to GeeksForGeeks");
}

void main() {
    // Calling the function
    GFG();
}
```

**Output:** `Welcome to GeeksForGeeks`

## Optional Parameters

Dart supports three parameter types:

| Parameter Type | Description |
|---|---|
| Optional Positional | Enclosed in square brackets `[ ]` |
| Optional Named | Enclosed in curly braces `{ }` |
| Default Values | Parameters assigned default values |

### Optional Parameters Example

```dart
void gfg1(int g1, [ var g2 ]) {
    print("g1 is $g1");
    print("g2 is $g2");
}

void gfg2(int g1, { var g2, var g3 }) {
    print("g1 is $g1");
    print("g2 is $g2");
    print("g3 is $g3");
}

void gfg3(int g1, { int g2 : 12 }) {
    print("g1 is $g1");
    print("g2 is $g2");
}

void main() {
    print("Calling the function with optional parameter:");
    gfg1(01);

    print("Calling the function with Optional Named parameter:");
    gfg2(01, g3 : 12);

    print("Calling function with default valued parameter");
    gfg3(01);
}
```

**Output:**
```
Calling the function with optional parameter:
g1 is 1
g2 is null
Calling the function with Optional Named parameter:
g1 is 1
g2 is null
g3 is 12
Calling function with default valued parameter
g1 is 1
g2 is 12
```

## Recursive Functions

Recursive functions call themselves, eliminating repeated function invocations.

### Fibonacci Series Example

```dart
/// Computes the nth Fibonacci number.
int fibonacci(int n) {
    // This is recursive function as it calls itself
    return n < 2 ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

void main() {
    // input
    var i = 20;

    print('fibonacci($i) = ${fibonacci(i)}');
}
```

**Output (for input 20):** `fibonacci(20) = 6765`

## Lambda Functions

Lambda functions (arrow functions) provide shorthand syntax for single-expression functions.

> But you should note that with lambda function you can return value for only one expression.

### Lambda Function Example

```dart
// Lambda function in Dart
void gfg() => print("Welcome to GeeksforGeeks");

void main() {
    // Calling Lambda function
    gfg();
}
```

**Output:** `Welcome to GeeksforGeeks`

## Key Concepts

- Functions improve code reusability and program organization
- Return types must be explicitly declared
- Optional parameters enhance function flexibility
- Recursive functions enable elegant solutions for certain problems
- Lambda syntax reduces boilerplate for simple operations

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-programming-functions/
- **Fetched**: 2026-01-27
