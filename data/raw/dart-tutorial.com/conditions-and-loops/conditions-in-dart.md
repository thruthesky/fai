# Conditions in Dart

## Overview

"When you write a computer program, you need to be able to tell the computer what to do in different situations." Conditions enable developers to control program flow in Dart by executing specific code blocks when particular situations are true.

## Types of Conditions

Dart supports four primary conditional structures:

- If Condition
- If-Else Condition
- If-Else-If Condition
- Switch case

## If Condition

The if statement is the most straightforward method for controlling program flow. It executes a code block when a given condition evaluates to true.

### Syntax

```dart
if(condition) {
    Statement 1;
    Statement 2;
    .
    .
    Statement n;
}
```

### Example

```dart
void main() {
    var age = 20;

    if(age >= 18){
      print("You are voter.");
    }
}
```

**Output:** `You are voter.`

## If-Else Condition

"If the result of the condition is true, then the body of the if-condition is executed. Otherwise, the body of the else-condition is executed."

### Syntax

```dart
if(condition){
    statements;
} else {
    statements;
}
```

### Example

```dart
void main(){
    int age = 12;
    if(age >= 18){
        print("You are voter.");
    } else {
        print("You are not voter.");
    }
}
```

**Output:** `You are not voter.`

### Boolean-Based Conditions

```dart
void main(){
    bool isMarried = false;
    if(isMarried){
        print("You are married.");
    } else {
        print("You are single.");
    }
}
```

**Output:** `You are single.`

## If-Else-If Condition

When handling multiple conditions, use if-else-if chains. This structure accommodates more than two conditional branches.

### Syntax

```dart
if(condition1){
    statements1;
} else if(condition2){
    statements2;
} else if(condition3){
    statements3;
}
.
.
else {
    statementsN;
}
```

### Example

```dart
void main() {
  int noOfMonth = 5;

  if (noOfMonth == 1) {
    print("The month is jan");
  } else if (noOfMonth == 2) {
    print("The month is feb");
  } else if (noOfMonth == 3) {
    print("The month is march");
  } else if (noOfMonth == 4) {
    print("The month is april");
  } else if (noOfMonth == 5) {
    print("The month is may");
  } else if (noOfMonth == 6) {
    print("The month is june");
  } else if (noOfMonth == 7) {
    print("The month is july");
  } else if (noOfMonth == 8) {
    print("The month is aug");
  } else if (noOfMonth == 9) {
    print("The month is sep");
  } else if (noOfMonth == 10) {
    print("The month is oct");
  } else if (noOfMonth == 11) {
    print("The month is nov");
  } else if (noOfMonth == 12) {
    print("The month is dec");
  } else {
    print("Invalid option given.");
  }
}
```

**Output:** `The month is may`

## Finding the Greatest Number Among Three

```dart
void main(){
    int num1 = 1200;
    int num2 = 1000;
    int num3 = 150;

    if(num1 > num2 && num1 > num3){
        print("Num 1 is greater: i.e $num1");
    }
    if(num2 > num1 && num2 > num3){
       print("Num2 is greater: i.e $num2");
    }
    if(num3 > num1 && num3 > num2){
        print("Num3 is greater: i.e $num3");
    }
}
```

**Output:** `Num 1 is greater: i.e 1200`

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/conditions-in-dart/
- **Fetched**: 2026-01-27
