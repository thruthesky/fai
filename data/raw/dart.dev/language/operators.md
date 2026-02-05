# Dart Operators - Complete Guide

## Overview

Dart supports a comprehensive set of operators with specific precedence and associativity rules. Operators can be implemented as class members, enabling custom behavior for user-defined types.

## Operator Precedence and Associativity

The language defines operators in a hierarchy from highest to lowest precedence:

**Highest Precedence:**
- Unary postfix: `++`, `--`, `()`, `[]`, `?.`, `.`, `!`
- Unary prefix: `-expr`, `!expr`, `~expr`, `++expr`, `--expr`, `await`

**Middle Levels:**
- Multiplicative: `*`, `/`, `%`, `~/` (left-associative)
- Additive: `+`, `-` (left-associative)
- Shift: `<<`, `>>`, `>>>` (left-associative)
- Bitwise: `&`, `^`, `|` (left-associative)
- Relational/Type test: `<`, `>`, `<=`, `>=`, `as`, `is`, `is!`
- Equality: `==`, `!=`
- Logical AND: `&&` (left-associative)
- Logical OR: `||` (left-associative)
- If-null: `??` (left-associative)

**Lowest Precedence:**
- Conditional: `? :`  (right-associative)
- Cascade: `..`, `?..` (left-associative)
- Assignment: `=`, `+=`, `-=`, etc. (right-associative)

## Arithmetic Operators

| Operator | Purpose |
|----------|---------|
| `+` | Addition |
| `-` | Subtraction |
| `-expr` | Negation (unary minus) |
| `*` | Multiplication |
| `/` | Division (returns double) |
| `~/` | Integer division |
| `%` | Modulo (remainder) |

**Example Usage:**
```dart
assert(2 + 3 == 5);
assert(5 / 2 == 2.5);
assert(5 ~/ 2 == 2);
assert(5 % 2 == 1);
```

### Increment and Decrement

- `++var` / `--var`: Modify before expression evaluation
- `var++` / `var--`: Modify after expression evaluation

```dart
a = 0;
b = ++a; // a becomes 1, b = 1
a = 0;
b = a++; // b = 0, a becomes 1
```

## Equality and Relational Operators

| Operator | Meaning |
|----------|---------|
| `==` | Equality (handles null correctly) |
| `!=` | Inequality |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal |
| `<=` | Less than or equal |

The equality operator `==` implements null-safe comparison: returns true if both operands are null, false if only one is null, otherwise invokes the `==` method on the first operand.

```dart
assert(2 == 2);
assert(3 > 2);
assert(2 <= 3);
```

For exact object identity, use the `identical()` function instead.

## Type Test Operators

| Operator | Purpose |
|----------|---------|
| `is` | Type check (returns boolean) |
| `is!` | Negative type check |
| `as` | Type casting/conversion |

```dart
if (employee is Person) {
  employee.firstName = 'Bob';
}

(employee as Person).firstName = 'Bob'; // Throws if not Person
```

## Assignment Operators

**Basic Assignment:**
```dart
a = value;
```

**Null-Coalescing Assignment:**
```dart
b ??= value; // Assign only if b is null
```

**Compound Assignment:**
Combines an operation with assignment: `a += b` equals `a = a + b`

Available compound operators: `*=`, `/=`, `%=`, `+=`, `-=`, `&=`, `^=`, `|=`, `<<=`, `>>=`, `>>>=`, `~/=`

```dart
var a = 2;
a *= 3; // a = a * 3 = 6
```

## Logical Operators

| Operator | Function |
|----------|----------|
| `!expr` | Logical NOT (inversion) |
| `\|\|` | Logical OR (short-circuits on true) |
| `&&` | Logical AND (short-circuits on false) |

```dart
if (!done && (col == 0 || col == 3)) {
  // Execute if not done AND column is 0 or 3
}
```

## Bitwise and Shift Operators

| Operator | Operation |
|----------|-----------|
| `&` | Bitwise AND |
| `\|` | Bitwise OR |
| `^` | Bitwise XOR |
| `~expr` | Bitwise complement (flip all bits) |
| `<<` | Shift left |
| `>>` | Arithmetic shift right |
| `>>>` | Unsigned shift right (Dart 2.14+) |

```dart
final value = 0x22;
final bitmask = 0x0f;

assert((value & bitmask) == 0x02);   // AND
assert((value | bitmask) == 0x2f);   // OR
assert((value ^ bitmask) == 0x2d);   // XOR
assert((value << 4) == 0x220);       // Left shift
assert((value >> 4) == 0x02);        // Right shift
assert((value >>> 4) == 0x02);       // Unsigned right shift
```

**Note:** Bitwise operations with large or negative numbers may differ between platforms.

## Conditional Expressions

**Ternary Conditional:**
```dart
condition ? expressionIfTrue : expressionIfFalse

var visibility = isPublic ? 'public' : 'private';
```

**Null-Coalescing (If-Null):**
```dart
expr1 ?? expr2  // Returns expr1 if non-null, otherwise expr2

String playerName(String? name) => name ?? 'Guest';
```

## Cascade Notation

The cascade operator (`..`) enables method chaining on a single object without reassignment:

```dart
var paint = Paint()
  ..color = Colors.black
  ..strokeCap = StrokeCap.round
  ..strokeWidth = 5.0;

// Equivalent to:
var paint = Paint();
paint.color = Colors.black;
paint.strokeCap = StrokeCap.round;
paint.strokeWidth = 5.0;
```

**Null-Safe Cascade (`?..`):**
Stops cascade operations if the object is null:

```dart
document.querySelector('#confirm')
  ?..textContent = 'Confirm'
  ..classList.add('important')
  ..onClick.listen((e) => window.alert('Confirmed!'))
  ..scrollIntoView();
```

**Important:** Cascades only work on functions returning actual objects, not void.

```dart
var sb = StringBuffer();
sb.write('foo')
  ..write('bar'); // ERROR: write() returns void
```

## Spread Operators

The spread operators (`...`, `...?`) unpack collection values into another collection. They're part of collection literal syntax, not standalone operators.

```dart
[...a + b]  // Unpacks result of a + b
```

Learn more on the Collections documentation page.

## Other Operators

| Operator | Name | Purpose |
|----------|------|---------|
| `()` | Function application | Represents function invocation |
| `[]` | Subscript access | Accesses collection element: `list[0]` |
| `?[]` | Conditional subscript | Safe access: `list?[0]` returns null if list is null |
| `.` | Member access | Accesses property: `obj.property` |
| `?.` | Conditional member access | Safe access: `obj?.property` returns null if obj is null |
| `!` | Non-null assertion | Forces non-nullable type: `obj!.property` throws if null |

## Key Takeaways

- "Operator precedence is an approximation; consult the language specification for authoritative behavior."
- Left-associative operators evaluate left-to-right; right-associative operators evaluate right-to-left
- Cascades provide fluent API design for object configuration
- Type test operators (`is`, `is!`, `as`) enable safe runtime type handling
- Null-aware operators (`?.`, `?[]`, `??`, `??=`) prevent null reference exceptions
