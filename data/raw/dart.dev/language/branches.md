# Branches in Dart

## Overview

This documentation page teaches developers how to control code flow using Dart's branching constructs: `if` statements, `if-case` statements, and `switch` statements/expressions.

## If Statements

Basic conditional execution uses `if` with optional `else` clauses. The condition must evaluate to a boolean:

```dart
if (isRaining()) {
  you.bringRainCoat();
} else if (isSnowing()) {
  you.wearJacket();
} else {
  car.putTopDown();
}
```

## If-Case Statements

The `if-case` construct matches a value against a pattern, executing the branch only when the pattern matches:

```dart
if (pair case [int x, int y]) return Point(x, y);
```

Variables defined by the pattern become available within the branch scope.

```dart
if (pair case [int x, int y]) {
  print('Was coordinate array $x,$y');
} else {
  throw FormatException('Invalid coordinates.');
}
```

**Note:** Requires language version 3.0 or later.

## Switch Statements

A `switch` statement evaluates an expression against multiple case patterns. Non-empty cases automatically jump to completion without requiring `break` statements:

```dart
var command = 'OPEN';
switch (command) {
  case 'CLOSED':
    executeClosed();
  case 'PENDING':
    executePending();
  case 'APPROVED':
    executeApproved();
  case 'DENIED':
    executeDenied();
  case 'OPEN':
    executeOpen();
  default:
    executeUnknown();
}
```

Empty cases fall through to subsequent cases, and labeled `continue` statements enable non-sequential fallthrough:

```dart
switch (command) {
  case 'OPEN':
    executeOpen();
    continue newCase;
  case 'DENIED':
  case 'CLOSED':
    executeClosed();
  newCase:
  case 'PENDING':
    executeNowClosed();
}
```

## Switch Expressions

These produce values based on matching cases and work wherever expressions are allowed:

```dart
token = switch (charCode) {
  slash || star || plus || minus => operator(charCode),
  comma || semicolon => punctuation(charCode),
  >= digit0 && <= digit9 => number(),
  _ => throw FormatException('Invalid'),
};
```

Key differences from statements:
- Cases omit the `case` keyword
- Bodies use single expressions instead of statements
- Cases separate patterns and bodies with `=>`
- Cases separate from each other with commas
- Default cases use only `_`

**Note:** Requires language version 3.0 or later.

## Exhaustiveness Checking

The compiler reports errors when a switch might miss possible values. Default cases cover all possibilities:

```dart
switch (nullableBool) {
  case true:
    print('yes');
  case false:
    print('no');
  // Missing case for null
}
```

Sealed classes enable exhaustiveness checking without defaults:

```dart
sealed class Shape {}

class Square implements Shape {
  final double length;
  Square(this.length);
}

class Circle implements Shape {
  final double radius;
  Circle(this.radius);
}

double calculateArea(Shape shape) => switch (shape) {
  Square(length: var l) => l * l,
  Circle(radius: var r) => math.pi * r * r,
};
```

## Guard Clauses

The `when` keyword adds optional boolean conditions after patterns in `case`, `if case`, or switch expressions:

```dart
switch (something) {
  case somePattern when condition:
    body;
}

var value = switch (something) {
  somePattern when condition => body,
};

if (something case somePattern when condition) {
  body;
}
```

Guards evaluate after pattern matching succeeds. Failed guards progress to the next case rather than exiting the switch.
