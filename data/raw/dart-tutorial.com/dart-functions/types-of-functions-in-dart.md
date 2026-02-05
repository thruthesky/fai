# Types of Functions in Dart

## Overview

"Functions are the block of code that performs a specific task." Dart supports four primary function categories based on parameters and return values.

## Function Categories

The four main types are:
- No Parameter And No Return Type
- Parameter And No Return Type
- No Parameter And Return Type
- Parameter And Return Type

## 1. Function With No Parameter And No Return Type

These functions execute code without accepting inputs or returning values.

### Example 1
```dart
void main() {
  printName();
}

void printName() {
  print("My name is John Doe.");
}
```

**Output:**
```
My name is John Doe.
```

The `void` keyword indicates no return type, and empty parentheses show no parameters are accepted.

### Example 2
```dart
void main() {
  print("Function With No Parameter and No Return Type");
  printPrimeMinisterName();
}

void printPrimeMinisterName() {
  print("John Doe.");
}
```

**Output:**
```
Function With No Parameter and No Return Type
John Doe.
```

## 2. Function With Parameter And No Return Type

These functions accept input parameters but don't return values.

### Example 1
```dart
void main() {
  printName("John");
}

void printName(String name) {
  print("Welcome, ${name}.");
}
```

**Output:**
```
Welcome, John.
```

### Example 2
```dart
void add(int a, int b) {
  int sum = a + b;
  print("The sum is $sum");
}

void main() {
  int num1 = 10;
  int num2 = 20;
  add(num1, num2);
}
```

**Output:**
```
The sum is 30
```

## 3. Function With No Parameter And Return Type

These functions produce output values without accepting parameters.

### Example 1
```dart
void main() {
  String name = primeMinisterName();
  print("The Name from function is $name.");
}

String primeMinisterName() {
  return "John Doe";
}
```

**Output:**
```
The name from function is John Doe
```

The return type `String` precedes the function name, indicating the value it produces.

### Example 2
```dart
void main() {
  int personAge = 17;
  if (personAge >= voterAge()) {
    print("You can vote.");
  } else {
    print("Sorry, you can't vote.");
  }
}

int voterAge() {
  return 18;
}
```

**Output:**
```
Sorry, you can't vote.
```

## 4. Function With Parameter And Return Type

These functions accept parameters and return computed values.

### Example 1
```dart
int add(int a, int b) {
  int sum = a + b;
  return sum;
}

void main() {
  int num1 = 10;
  int num2 = 20;
  int total = add(num1, num2);
  print("The sum is $total.");
}
```

**Output:**
```
The sum is 30.
```

### Example 2
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

**Output:**
```
The simple interest is 450.0.
```

## Complete Example

```dart
int add(int a, int b) {
  var total;
  total = a + b;
  return total;
}

void mul(int a, int b) {
  var total;
  total = a * b;
  print("Multiplication is : $total");
}

String greet() {
  String greet = "Welcome";
  return greet;
}

void greetings() {
  print("Hello World!!!");
}

void main() {
  var total = add(2, 3);
  print("Total sum: $total");
  mul(2, 3);
  var greeting = greet();
  print("Greeting: $greeting");
  greetings();
}
```

**Output:**
```
Total sum: 5
Multiplication is : 6
Greeting: Welcome
Hello World!!!
```

## Key Note

"void is used for no return type as it is a non value-returning function."

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/types-of-functions-in-dart/
- **Fetched**: 2026-01-27
