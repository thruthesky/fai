# Ternary Operator in Dart

## Overview

The ternary operator functions as a concise substitute for if-else statements. It evaluates a boolean condition and selects one of two values based on the result.

## Syntax

```
condition ? exprIfTrue : exprIfFalse
```

**Key Point:** This operator accepts a condition and returns one of two values depending on whether the condition evaluates to true or false.

## Ternary Operator vs If-Else

### Traditional If-Else Approach

```dart
void main() {
  int num1 = 10;
  int num2 = 15;
  int max = 0;
  if(num1 > num2){
    max = num1;
  } else {
    max = num2;
  }
  print("The greatest number is $max");
}
```

**Output:** `The greatest number is 15`

### Using Ternary Operator

```dart
void main() {
  int num1 = 10;
  int num2 = 15;
  int max = (num1 > num2) ? num1 : num2;
  print("The greatest number is $max");
}
```

**Output:** `The greatest number is 15`

The ternary approach reduces code length and improves readability significantly.

## Example 2: String Selection

```dart
void main() {
  var selection = 2;
  var output = (selection == 2) ? 'Apple' : 'Banana';
  print(output);
}
```

**Output:** `Apple`

## Example 3: Voter Eligibility Check

```dart
void main() {
  var age = 18;
  var check = (age >= 18) ? 'You ara a voter.' : 'You are not a voter.';
  print(check);
}
```

**Output:** `You ara a voter.`

## Challenge Exercise

Create an integer variable for age. Write a ternary statement that outputs "Teenager" if age falls between 13-19, otherwise "Not Teenager".

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/ternary-operator-in-dart/
- **Fetched**: 2026-01-27
