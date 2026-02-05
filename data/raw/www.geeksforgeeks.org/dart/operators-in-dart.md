# Operators in Dart - Complete Tutorial

## Introduction

The operators are special symbols used to perform certain operations on operands. Dart provides numerous built-in operators for various functions, such as the '+' symbol for addition. These operators work on one or two operands.

## Operator Precedence Table

| Description | Operator | Associativity |
|---|---|---|
| Unary postfix | expr++ expr-- () [] ?[] . ?. ! | Right |
| Unary prefix | -expr !expr ~expr ++expr --expr await expr | Right |
| Multiplicative | * / % ~/ | Left |
| Additive | + - | Left |
| Shift | << >> >>> | Left |
| Bitwise AND | & | Left |
| Bitwise XOR | ^ | Left |
| Bitwise OR | \| | Left |
| Relational/Type test | >= > <= < as is is! | Left |
| Equality | == != | Left |
| Logical AND | && | Left |
| Logical OR | \|\| | Left |
| If-null | ?? | Left |
| Conditional | expr ? expr2 : expr3 | Right |
| Cascade | .. ?.. | Left |
| Assignment | = *= /= += -= &= ^= etc. | Right |

---

## 1. Arithmetic Operators

These operators perform mathematical operations on two operands:

| Symbol | Name | Description |
|---|---|---|
| + | Addition | Adds two operands |
| - | Subtraction | Subtracts two operands |
| -expr | Unary Minus | Reverses the sign of an expression |
| * | Multiplication | Multiplies two operands |
| / | Division | Divides two operands |
| ~/ | Integer Division | Returns quotient as integer |
| % | Modulus | Returns remainder |

### Example

```dart
void main() {
  int a = 2;
  int b = 3;

  var c = a + b;
  print("Sum (a + b) = $c");

  var d = a - b;
  print("Difference (a - b) = $d");

  var e = -d;
  print("Negation -(a - b) = $e");

  var f = a * b;
  print("Product (a * b) = $f");

  var g = b / a;
  print("Division (b / a) = $g");

  var h = b ~/ a;
  print("Quotient (b ~/ a) = $h");

  var i = b % a;
  print("Remainder (b % a) = $i");
}
```

### Output
```
Sum (a + b) = 5
Difference (a - b) = -1
Negation -(a - b) = 1
Product (a * b) = 6
Division (b / a) = 1.5
Quotient (b ~/ a) = 1
Remainder (b % a) = 1
```

---

## 2. Relational Operators

These operators compare two operands and return boolean values:

| Symbol | Name | Description |
|---|---|---|
| > | Greater than | Checks if left operand is larger |
| < | Less than | Checks if left operand is smaller |
| >= | Greater than or equal | Checks if left is greater or equal |
| <= | Less than or equal | Checks if left is less or equal |
| == | Equal to | Checks operand equality |
| != | Not equal to | Checks operand inequality |

### Example

```dart
void main() {
  int a = 2;
  int b = 3;

  var c = a > b;
  print("a is greater than b (a > b) : $c");

  var d = a < b;
  print("a is smaller than b (a < b) : $d");

  var e = a >= b;
  print("a is greater than or equal to b (a >= b) : $e");

  var f = a <= b;
  print("a is smaller than or equal to b (a <= b) : $f");

  var g = b == a;
  print("a and b are equal (b == a) : $g");

  var h = b != a;
  print("a and b are not equal (b != a) : $h");
}
```

### Output
```
a is greater than b (a > b) : false
a is smaller than b (a < b) : true
a is greater than or equal to b (a >= b) : false
a is smaller than or equal to b (a <= b) : true
a and b are equal (b == a) : false
a and b are not equal (b != a) : true
```

**Note:** The == operator can't be used to check if the object is same. Use the `identical()` function instead.

---

## 3. Type Test Operators

### is and is! Operators

These operators check if an object has a specific type:

| Symbol | Name | Description |
|---|---|---|
| is | Type check | Returns true if object matches type |
| is! | Negated type check | Returns false if object matches type |

### Example

```dart
void main() {
  String a = 'GFG';
  double b = 3.3;

  print(a is String);
  print(b is !int);
}
```

### Output
```
true
true
```

### as Operator (Type Casting)

The as Operator is used for Typecasting. It performs runtime casting; if invalid, it throws an error.

```dart
void main() {
  dynamic value = "Hello";
  String str = value as String;
  print(str);
}
```

### Output
```
Hello
```

---

## 4. Bitwise Operators

These operators perform bitwise operations on integer operands:

| Symbol | Name | Description |
|---|---|---|
| & | Bitwise AND | AND operation on bits |
| \| | Bitwise OR | OR operation on bits |
| ^ | Bitwise XOR | XOR operation on bits |
| ~ | Bitwise NOT | NOT operation on bits |
| << | Left shift | Shifts bits left, fills with 0 from right |
| >> | Right shift | Shifts bits right, preserves sign |
| >>> | Unsigned right shift | Shifts bits right, fills with 0 |

### Example

```dart
void main() {
  int a = 5;
  int b = 7;

  var c = a & b;
  print("a & b : $c");

  var d = a | b;
  print("a | b : $d");

  var e = a ^ b;
  print("a ^ b : $e");

  var f = ~a;
  print("~a : $f");

  var g = a << b;
  print("a << b : $g");

  var h = a >> b;
  print("a >> b : $h");

  var i = -a >>> b;
  print("-a >>> b : $i");
}
```

### Output
```
a & b : 5
a | b : 7
a ^ b : 2
~a : 4294967290
a << b : 640
a >> b : 0
-a >>> b : 33554431
```

---

## 5. Assignment Operators

These operators assign values to variables:

| Symbol | Name | Description |
|---|---|---|
| = | Assignment | Assigns value to variable |
| ??= | Null-coalescing assignment | Assigns only if variable is null |

### Example

```dart
void main() {
  int a = 5;
  int b = 7;

  var c = a * b;
  print("assignment operator used c = a*b so now c = $c\n");

  var d;
  d ??= a + b;
  print("Assigning value only if d is null");
  print("d??= a+b so d = $d \n");

  d ??= a - b;
  print("Assigning value only if d is null");
  print("d??= a-b so d = $d");
  print("As d was not null value was not updated");
}
```

### Output
```
assignment operator used c = a*b so now c = 35

Assigning value only if d is null
d??= a+b so d = 12

Assigning value only if d is null
d??= a-b so d = 12
As d was not null value was not updated
```

### Compound Assignment Operators

Combine operators with assignment for concise code:

```dart
a += 1;  // Same as: a = a + 1;
a -= 1;  // Same as: a = a - 1;
a *= 2;  // Same as: a = a * 2;
a /= 2;  // Same as: a = a / 2;
```

---

## 6. Logical Operators

These operators combine boolean conditions:

| Symbol | Name | Description |
|---|---|---|
| && | AND | Returns true if both conditions are true |
| \|\| | OR | Returns true if at least one is true |
| ! | NOT | Reverses boolean value |

### Example

```dart
void main() {
  int a = 5;
  int b = 7;

  bool c = a > 10 && b < 10;
  print(c);

  bool d = a > 10 || b < 10;
  print(d);

  bool e = !(a > 10);
  print(e);
}
```

### Output
```
false
true
true
```

**Important Note:** Logical operator can only be application to boolean expression. Non-zero numbers aren't automatically true in Dart.

### Incorrect Usage Example

```dart
void main() {
  int a = 5;
  int b = 7;
  print(a && b);  // Error: int cannot be assigned to bool
}
```

### Correct Usage Example

```dart
void main() {
  var a = true;
  var b = false;

  print("a: $a , b: $b\n");
  print("a && b = ${a&&b}");
  print("a || b = ${a||b}");
  print("!a = ${!a}");
}
```

### Output
```
a: true , b: false

a && b = false
a || b = true
!a = false
```

---

## 7. Conditional Operators

These operators handle conditional logic:

| Symbol | Name | Description |
|---|---|---|
| condition ? expr1 : expr2 | Ternary | Returns expr1 if true, expr2 if false |
| expr1 ?? expr2 | Null-coalescing | Returns expr1 if non-null, else expr2 |

### Example

```dart
void main() {
  int a = 5;

  var c = (a < 10) ? "Statement is Correct, Geek" :
          "Statement is Wrong, Geek";
  print(c);

  int? n;
  var d = n ?? "n has Null value";
  print(d);

  n = 10;
  d = n;
  print(d);
}
```

### Output
```
Statement is Correct, Geek
n has Null value
10
```

**Note:** Declaring `int? n` indicates the variable can hold either an integer or null. The `?` denotes nullability.

---

## 8. Cascade Notation Operators

These operators allow multiple operations on the same object:

| Symbol | Name | Description |
|---|---|---|
| .. | Cascade | Performs multiple method calls on one object |
| ..? | Null-safe cascade | Operates only if object is non-null |

### Example

```dart
class GFG {
  int? a;
  int? b;

  void set(int x, int y) {
    this.a = x;
    this.b = y;
  }

  void add() {
    if (a != null && b != null) {
      var z = a! + b!;
      print(z);
    } else {
      print("Values are not initialized.");
    }
  }
}

void main() {
  GFG geek1 = GFG();
  GFG geek2 = GFG();

  // Without cascade notation
  geek1.set(1, 2);
  geek1.add();

  // Using cascade notation
  geek2
    ..set(3, 4)
    ..add();
}
```

### Output
```
3
7
```

---

## Summary

Dart provides a comprehensive set of operators for:
- **Arithmetic**: Mathematical calculations
- **Relational**: Comparisons
- **Type Testing**: Type checking and casting
- **Bitwise**: Low-level bit operations
- **Assignment**: Value assignment
- **Logical**: Boolean combinations
- **Conditional**: Ternary and null-coalescing logic
- **Cascade**: Chaining multiple operations

Understanding operator precedence and proper usage ensures clean, efficient Dart code.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/operators-in-dart/
- **Fetched**: 2026-01-27
