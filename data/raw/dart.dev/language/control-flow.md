# Loops in Dart

This documentation page covers control flow mechanisms for iterating through code in Dart.

## For Loops

Standard `for` loops work with traditional counter syntax:

```dart
var message = StringBuffer('Dart is fun');
for (var i = 0; i < 5; i++) {
  message.write('!');
}
```

A key distinction: "Closures inside of Dart's `for` loops capture the _value_ of the index," preventing a common JavaScript pitfall where nested functions reference the final counter value rather than each iteration's value.

### For-In Loops

When iteration counters aren't needed, use the cleaner `for-in` construct:

```dart
for (var candidate in candidates) {
  candidate.interview();
}
```

Variables defined in the loop body are local to that iteration. Pattern destructuring is also supported:

```dart
for (final Candidate(:name, :yearsExperience) in candidates) {
  print('$name has $yearsExperience of experience.');
}
```

### forEach() Method

Collections support `forEach()` for functional-style iteration:

```dart
var collection = [1, 2, 3];
collection.forEach(print); // 1 2 3
```

## While and Do-While Loops

`while` loops evaluate conditions before executing:

```dart
while (!isDone()) {
  doSomething();
}
```

`do-while` loops evaluate conditions after executing:

```dart
do {
  printLine();
} while (!atEndOfPage());
```

## Break and Continue

`break` terminates looping immediately:

```dart
while (true) {
  if (shutDownRequested()) break;
  processIncomingRequests();
}
```

`continue` advances to the next iteration:

```dart
for (int i = 0; i < candidates.length; i++) {
  var candidate = candidates[i];
  if (candidate.yearsExperience < 5) {
    continue;
  }
  candidate.interview();
}
```

Alternatively, functional approaches work well:

```dart
candidates
    .where((c) => c.yearsExperience >= 5)
    .forEach((c) => c.interview());
```

## Labels

Labels provide targeted control over nested structures. Format: `labelName:` preceding a statement.

**Label with `break`:** Terminates execution of the labeled statement, useful for exiting outer loops within nested structures.

**Label with `continue`:** Skips remaining iterations of the labeled loop and proceeds to the next iteration.

### For Loop Examples

Using `break` to exit nested loops:

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

Using `continue` to skip to the next outer iteration:

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

### While and Do-While Examples

Similar label patterns apply to `while` and `do-while` constructs, allowing fine-grained control over loop termination and continuation in nested scenarios.

---

**Document Source:** Dart.dev Language Documentation
