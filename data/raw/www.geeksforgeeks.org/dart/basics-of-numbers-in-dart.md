# Basics of Numbers in Dart - Complete Tutorial

## Introduction

Dart, like other programming languages, treats numerical values as Number objects. The number data type in Dart holds numeric values and can be classified into two main categories.

## Number Types in Dart

### 1. **int (Integer)**

The `int` data type represents whole numbers without decimal points.

**Declaration syntax:**
```dart
int var_name;
```

**Example:**
```dart
// Declaring an integer variable
int age = 25;
```

### 2. **double (Floating-Point Number)**

The `double` data type represents 64-bit floating-point numbers with decimal precision.

**Declaration syntax:**
```dart
double var_name;
```

**Example:**
```dart
// Declaring a double variable
double pi = 3.1415;
```

### Complete Usage Example

```dart
void main() {
  // declare an integer
  int num1 = 2;

  // declare a double value
  double num2 = 1.5;

  // print the values
  print(num1);
  print(num2);
}
```

**Output:**
```
2
1.5
```

**Important Note:** The `num` is actually the superclass of `int` and `double`, meaning both extend this parent class for type flexibility.

---

## Parsing in Dart

The `parse()` function converts valid numeric strings into numbers. Invalid strings throw a `FormatException`.

```dart
void main() {
  // Converts "1" into a numeric value (int)
  var a1 = num.parse("1");

  // Converts "2.34" into a numeric value (double)
  var b1 = num.parse("2.34");

  // Performing addition operation on parsed numbers
  var c1 = a1 + b1;

  // Printing the result
  print("Sum = ${c1}");
}
```

**Output:**
```
Sum = 3.34
```

---

## Properties of Numbers

| Property | Description |
|----------|-------------|
| `hashCode` | Returns the hash code of the given number |
| `isFinite` | Returns `true` if the number is finite |
| `isInfinite` | Returns `true` if the number is infinite |
| `isNaN` | Returns `true` if the number is not a valid numeric value |
| `isNegative` | Returns `true` if the number is negative |
| `sign` | Returns -1 (negative), 0 (zero), or 1 (positive) |
| `isEven` | Returns `true` if the number is even |
| `isOdd` | Returns `true` if the number is odd |

### Properties Implementation Example

```dart
void main() {
  int num1 = 10;
  double num2 = -5.5;
  double num3 = double.infinity;
  double num4 = 0 / 0; // NaN

  print("Hash code of num1: ${num1.hashCode}");
  print("Is num1 finite? ${num1.isFinite}"); // true
  print("Is num3 infinite? ${num3.isInfinite}"); // true
  print("Is num4 NaN? ${num4.isNaN}"); // true
  print("Is num2 negative? ${num2.isNegative}"); // true
  print("Sign of num1: ${num1.sign}"); // 1
  print("Is num1 even? ${num1.isEven}"); // true
  print("Is num1 odd? ${num1.isOdd}"); // false
}
```

---

## Methods of Numbers

| Method | Description |
|--------|-------------|
| `abs()` | Returns the absolute value |
| `ceil()` | Returns the ceiling value (smallest integer >= number) |
| `floor()` | Returns the floor value (largest integer <= number) |
| `compareTo()` | Compares value with another number |
| `remainder()` | Returns truncated remainder after division |
| `round()` | Returns the rounded value |
| `toDouble()` | Converts to double representation |
| `toInt()` | Converts to integer representation |
| `toString()` | Converts to String representation |
| `truncate()` | Returns integer after discarding fraction digits |

### Methods Implementation Example

```dart
void main() {
  double number = -12.75;
  int intNumber = 15;

  print('Absolute Value: ${number.abs()}'); // 12.75
  print('Ceiling Value: ${number.ceil()}'); // -12
  print('Floor Value: ${number.floor()}'); // -13
  print('Compare To (10): ${number.compareTo(10)}'); // -1
  print('Remainder when divided by 5: ${number.remainder(5)}'); // -2.75
  print('Rounded Value: ${number.round()}'); // -13
  print('Integer to Double: ${intNumber.toDouble()}'); // 15.0
  print('Double to Integer: ${number.toInt()}'); // -12
  print('Number as String: ${number.toString()}'); // -12.75
  print('Truncated Value: ${number.truncate()}'); // -12
}
```

**Output:**
```
Absolute Value: 12.75
Ceiling Value: -12
Floor Value: -13
Compare To (10): -1
Remainder when divided by 5: -2.75
Rounded Value: -13
Integer to Double: 15.0
Double to Integer: -12
Number as String: -12.75
Truncated Value: -12
```

---

## Type Assignment Considerations

### Important Behavior:

You can assign `int` values to `double` variables, but not vice versa without conversion:

```dart
void main(){
  int x = 10;

  // Error: Cannot assign double to int
  x = 20.5;

  num y = 10;
  // No error with num type
  y = 10.5;

  double z = 10; // Allowed
  z = 10.5; // Also allowed
}
```

---

## Summary

Dart provides robust number handling through `int` and `double` types, both extending the `num` superclass. The language offers comprehensive properties for number inspection and methods for transformation. The `num.parse()` function enables string-to-number conversion, while numerous built-in methods facilitate mathematical operations and type conversions efficiently.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/basics-of-numbers-in-dart/
- **Fetched**: 2026-01-27
