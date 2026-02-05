# Operators in Dart

## Overview

Operators facilitate mathematical and logical computations on variables. Each operation employs a symbol—the operator—to indicate the type of action performed.

### Key Concepts

- **Operands**: Data values being processed
- **Operator**: Symbol indicating how operands are processed

*Example: In the expression `2 + 3`, the numbers are operands and `+` is the operator.*

## Types of Operators

Dart includes five primary operator categories:

1. Arithmetic Operators
2. Increment and Decrement Operators
3. Assignment Operators
4. Relational Operators
5. Logical Operators
6. Type Test Operators

---

## Arithmetic Operators

These perform fundamental mathematical operations like addition and subtraction.

| Symbol | Name | Purpose |
|--------|------|---------|
| `+` | Addition | Sum of operands |
| `-` | Subtraction | Difference of operands |
| `-expr` | Unary Minus | Reverses expression sign |
| `*` | Multiplication | Product of operands |
| `/` | Division | Result as double |
| `~/` | Integer Division | Result as integer |
| `%` | Modulus | Remainder after division |

### Example

```dart
void main() {
 int num1=10;
 int num2=3;

 int sum=num1+num2;       // 13
 int diff=num1-num2;      // 7
 int unaryMinus = -num1;  // -10
 int mul=num1*num2;       // 30
 double div=num1/num2;    // 3.333...
 int div2 =num1~/num2;    // 3
 int mod=num1%num2;       // 1

 print("The addition is $sum.");
 print("The subtraction is $diff.");
 print("The unary minus is $unaryMinus.");
 print("The multiplication is $mul.");
 print("The division is $div.");
 print("The integer division is $div2.");
 print("The modulus is $mod.");
}
```

---

## Increment and Decrement Operators

These modify values by one unit. Prefix operators increment/decrement before expression evaluation; postfix operators do so after.

| Symbol | Name | Behavior |
|--------|------|----------|
| `++var` | Pre Increment | Increases by 1; expression evaluates to new value |
| `--var` | Pre Decrement | Decreases by 1; expression evaluates to new value |
| `var++` | Post Increment | Increases by 1; expression evaluates to original value |
| `var--` | Post Decrement | Decreases by 1; expression evaluates to original value |

### Example

```dart
void main() {
 int num1=0;
 int num2=0;

 // pre increment
 num2 = ++num1;
 print("The value of num2 is $num2");  // 1

 num1 = 0;
 num2 = 0;

 // post increment
 num2 =  num1++;
 print("The value of num2 is $num2");  // 0
}
```

---

## Assignment Operators

These assign or modify variable values.

| Symbol | Function |
|--------|----------|
| `=` | Assign value |
| `+=` | Add and assign |
| `-=` | Subtract and assign |
| `*=` | Multiply and assign |
| `/=` | Divide and assign |

### Example

```dart
void main() {
  double age = 24;
  age+= 1;
  print("After Addition Age is $age");      // 25.0
  age-= 1;
  print("After Subtraction Age is $age");   // 24.0
  age*= 2;
  print("After Multiplication Age is $age"); // 48.0
  age/= 2;
  print("After Division Age is $age");      // 24.0
}
```

---

## Relational Operators

Also termed comparison operators, these evaluate relationships between values, returning boolean results.

| Symbol | Name | Purpose |
|--------|------|---------|
| `>` | Greater than | Checks which is larger |
| `<` | Less than | Checks which is smaller |
| `>=` | Greater than or equal | Compares larger/equal status |
| `<=` | Less than or equal | Compares smaller/equal status |
| `==` | Equal to | Checks equality |
| `!=` | Not equal to | Checks inequality |

### Example

```dart
void main() {
 int num1=10;
 int num2=5;

 print(num1==num2);   // false
 print(num1<num2);    // false
 print(num1>num2);    // true
 print(num1<=num2);   // false
 print(num1>=num2);   // true
}
```

---

## Logical Operators

These combine or negate boolean conditions.

| Symbol | Purpose |
|--------|---------|
| `&&` | AND—true only if both conditions are true |
| `\|\|` | OR—true if at least one condition is true |
| `!` | NOT—inverts the boolean value |

### Example

```dart
void main(){
  int userid = 123;
  int userpin = 456;

  print((userid == 123) && (userpin== 456)); // true
  print((userid == 1213) && (userpin== 456)); // false
  print((userid == 123) || (userpin== 456)); // true
  print((userid == 1213) || (userpin== 456)); // true
  print((userid == 123) != (userpin== 456));  // false
}
```

---

## Type Test Operators

These examine object types during execution.

| Symbol | Name | Function |
|--------|------|----------|
| `is` | Is | Returns true if object matches type |
| `is!` | Is not | Returns true if object doesn't match type |

### Example

```dart
void main() {
  String value1 = "Dart Tutorial";
  int age = 10;

  print(value1 is String);   // true
  print(age is !int);        // false
}
```

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/operators-in-dart/
- **Fetched**: 2026-01-27
