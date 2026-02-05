# Loops in Dart

## Overview

Dart provides several loop constructs to control code flow: `for` loops, `while` loops, `do-while` loops, along with `break` and `continue` statements for managing iterations.

## For Loops

### Standard For Loop

A traditional `for` loop with initialization, condition, and increment:

```dart
var message = StringBuffer('Dart is fun');
for (var i = 0; i < 5; i++) {
  message.write('!');
}
```

**Key characteristic:** Closures in Dart's `for` loops capture the index value itself, not a reference. This differs from JavaScript behavior.

```dart
var callbacks = [];
for (var i = 0; i < 2; i++) {
  callbacks.add(() => print(i));
}

for (final c in callbacks) {
  c();
}
// Output: 0, then 1
```

### For-In Loop

When you don't need the iteration counter, use `for-in` with iterables like lists or sets:

```dart
for (var candidate in candidates) {
  candidate.interview();
}
```

The loop variable is local to each iteration and doesn't affect the original collection.

### Pattern-Based For-In Loop

Destructure values using patterns:

```dart
for (final Candidate(:name, :yearsExperience) in candidates) {
  print('$name has $yearsExperience of experience.');
}
```

### ForEach Method

Iterable classes provide a `forEach()` method:

```dart
var collection = [1, 2, 3];
collection.forEach(print); // 1 2 3
```

## While and Do-While Loops

### While Loop

Evaluates the condition before executing the loop body:

```dart
while (!isDone()) {
  doSomething();
}
```

### Do-While Loop

Evaluates the condition after executing the loop body, ensuring at least one iteration:

```dart
do {
  printLine();
} while (!atEndOfPage());
```

## Break and Continue

### Break Statement

Terminates loop execution immediately:

```dart
while (true) {
  if (shutDownRequested()) break;
  processIncomingRequests();
}
```

### Continue Statement

Skips the current iteration and proceeds to the next:

```dart
for (int i = 0; i < candidates.length; i++) {
  var candidate = candidates[i];
  if (candidate.yearsExperience < 5) {
    continue;
  }
  candidate.interview();
}
```

Alternatively, use functional approaches:

```dart
candidates
    .where((c) => c.yearsExperience >= 5)
    .forEach((c) => c.interview());
```

## Labels

Labels allow you to break out of or continue specific outer loops in nested structures. Format: `labelName:` preceding a statement.

### Break with Labels

```dart
outerLoop:
for (var i = 1; i <= 3; i++) {
  for (var j = 1; j <= 3; j++) {
    print('i = $i, j = $j');
    if (i == 2 && j == 2) {
      break outerLoop;
    }
  }
}
print('outerLoop exited');
```

**Output:** Exits both loops when condition is met.

### Continue with Labels

```dart
outerLoop:
for (var i = 1; i <= 3; i++) {
  for (var j = 1; j <= 3; j++) {
    if (i == 2 && j == 2) {
      continue outerLoop;
    }
    print('i = $i, j = $j');
  }
}
```

**Output:** Skips remaining inner iterations and advances the outer loop.

Labels work with `while` and `do-while` loops identically, allowing controlled navigation through nested loop structures.

---

**Document Source:** Dart.dev Language Documentation
