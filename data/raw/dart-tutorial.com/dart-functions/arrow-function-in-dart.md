# Arrow Function in Dart

## Overview

Dart provides a concise syntax for functions containing a single expression using the arrow operator. As the tutorial states: "The arrow function is represented by **=>** symbol. It is a shorthand syntax for any function that has only one expression."

## Syntax

```
returnType functionName(parameters...) => expression;
```

The arrow notation serves as shorthand for `{ return expr; }`, enabling more compact code.

## Example 1: Standard Function Approach

This demonstrates calculating simple interest without arrow syntax:

```dart
double calculateInterest(double principal, double rate, double time) {
  double interest = principal * rate * time / 100;
  return interest;
}

void main() {
  double principal = 5000;
  double time = 3;
  double rate = 3;

  double result = calculateInterest(principal, rate, time);
  print("The simple interest is $result.");
}
```

**Output:** `Simple interest is 450.0`

## Example 2: Arrow Function Implementation

The same calculation using arrow syntax:

```dart
double calculateInterest(double principal, double rate, double time) =>
    principal * rate * time / 100;

void main() {
  double principal = 5000;
  double time = 3;
  double rate = 3;

  double result = calculateInterest(principal, rate, time);
  print("The simple interest is $result.");
}
```

**Output:** `Simple interest is 450.0`

## Example 3: Multiple Arithmetic Operations

Arrow functions for basic mathematical operations:

```dart
int add(int n1, int n2) => n1 + n2;
int sub(int n1, int n2) => n1 - n2;
int mul(int n1, int n2) => n1 * n2;
double div(int n1, int n2) => n1 / n2;

void main() {
  int num1 = 100;
  int num2 = 30;

  print("The sum is ${add(num1, num2)}");
  print("The diff is ${sub(num1, num2)}");
  print("The mul is ${mul(num1, num2)}");
  print("The div is ${div(num1, num2)}");
}
```

**Output:**
```
The sum is 130
The diff is 70
The mul is 3000
The div is 3.3333333333333335
```

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/arrow-function-in-dart/
- **Fetched**: 2026-01-27
