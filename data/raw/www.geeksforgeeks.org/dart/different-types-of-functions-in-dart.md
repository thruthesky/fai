# Different Types of Functions in Dart

## Overview

The function is a set of statements that take inputs, do some specific computation, and produce output. Functions promote code reusability and help organize complex programs into manageable components.

## Four Function Types in Dart

### 1. No Arguments and No Return Type

This function type executes a task without accepting parameters or returning values.

```dart
void myName() {
  print("GeeksForGeeks");
}

void main() {
  print("This is the best website for developers:");
  myName();
}
```

**Output:**
```
This is the best website for developers:
GeeksForGeeks
```

The `void` keyword indicates no return value, and empty parentheses show no parameters are accepted.

### 2. No Arguments with Return Type

These functions return a value without requiring input parameters.

```dart
int myPrice() {
  int price = 0;
  return price;
}

void main() {
  int Price = myPrice();
  print("GeeksforGeeks is the best website"
      + " for developers which costs : ${Price}/-");
}
```

**Output:**
```
GeeksforGeeks is the best website for developers which costs : 0/-
```

The return type `int` specifies what data type the function produces.

### 3. With Arguments and No Return Type

Functions that process input but don't return results.

```dart
void myPrice(int price) {
  print(price);
}

void main() {
  print("GeeksforGeeks is the best website"
      + " for developers which costs : ");
  myPrice(0);
}
```

**Output:**
```
GeeksforGeeks is the best website for developers which costs : 0
```

The parentheses contain parameters the function uses during execution.

### 4. With Arguments and Return Type

The most comprehensive function type—accepting inputs and producing outputs.

```dart
int mySum(int firstNumber, int secondNumber) {
  return (firstNumber + secondNumber);
}

void main() {
  int additionOfTwoNumber = mySum(100, 500);
  print(additionOfTwoNumber);
}
```

**Output:**
```
600
```

## When to Use Each Type

- **No Arguments, No Return Type**: Execute standalone tasks
- **No Arguments, With Return Type**: Generate computed values independently
- **With Arguments, No Return Type**: Process data without producing results
- **With Arguments, With Return Type**: Transform input into calculated output

Selecting appropriate function types enables writing clean, reusable, and efficient Dart code.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/different-types-of-functions-in-dart/
- **Fetched**: 2026-01-27
