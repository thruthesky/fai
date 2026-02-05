# Functions in Dart

## Function In Dart

Functions are reusable blocks of code that perform specific tasks. They embody the principle of "DRY (Don't Repeat Yourself)" by eliminating code repetition throughout programs.

## Function Advantages

- Avoid Code Repetition
- Easy to divide the complex program into smaller parts
- Helps to write a clean code

## Syntax

```dart
returntype functionName(parameter1, parameter2, ...){
  // function body
}
```

**Return type**: Specifies the output type (void, String, int, double, etc.)

**Function Name**: Use lowerCamelCase convention (e.g., `printName()`)

**Parameters**: Inputs defined in parentheses; follow lowerCamelCase naming

## Example 1: Function That Prints Name

```dart
void printName(){
  print("My name is Raj Sharma. I am from function.");
}

void main(){
  printName();
}
```

**Output:**
```
My name is Raj Sharma. I am from function.
```

## Example 2: Function To Find Sum of Two Numbers

```dart
void add(int num1, int num2){
  int sum = num1 + num2;
  print("The sum is $sum");
}

void main(){
  add(10, 20);
}
```

**Output:**
```
The sum is 30
```

## Example 3: Function That Find Simple Interest

```dart
void calculateInterest(double principal, double rate, double time) {
  double interest = principal * rate * time / 100;
  print("Simple interest is $interest");
}

void main() {
  double principal = 5000;
  double time = 3;
  double rate = 3;
  calculateInterest(principal, rate, time);
}
```

**Output:**
```
Simple interest is 450.0
```

## Challenge

Create a function that finds a cube of numbers.

## Key Points

- In Dart, functions are objects
- Use lowerCamelCase naming convention for functions and parameters

## About lowerCamelCase

Names start with lowercase; subsequent words begin with uppercase (e.g., `num1`, `fullName`, `isMarried`).

## Function Parameters Vs Arguments

```dart
void add(int num1, int num2){
  int sum;
  sum = num1 + num2;
  print("The sum is $sum");
}

void main(){
  add(10, 20);
}
```

- **Parameters**: `num1` and `num2` in the function definition
- **Arguments**: `10` and `20` passed when calling the function
- Parameters define inputs; arguments are actual values supplied

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/functions-in-dart/
- **Fetched**: 2026-01-27
