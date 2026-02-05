# Break and Continue in Dart

## Dart Break and Continue

When working with loops, you may need to skip certain elements or terminate iteration without evaluating the condition. The break and continue statements serve these purposes.

## Break Statement

The break statement immediately exits a loop, transferring control outside of it.

**Syntax:**
```dart
break;
```

### Example 1: Break In Dart For Loop

This loop terminates when the counter reaches 5, even though the condition allows iteration up to 10.

```dart
void main() {
  for (int i = 1; i <= 10; i++) {
    if (i == 5) {
      break;
    }
    print(i);
  }
}
```

**Output:**
```
1
2
3
4
```

### Example 2: Break In Dart Negative For Loop

This descending loop stops when the value reaches 7.

```dart
void main() {
  for (int i = 10; i >= 1; i--) {
    if (i == 7) {
      break;
    }
    print(i);
  }
}
```

**Output:**
```
10
9
8
```

### Example 3: Break In Dart While Loop

The while loop exits when the counter equals 5.

```dart
void main() {
  int i = 1;
  while(i <= 10) {
    print(i);
    if (i == 5) {
      break;
    }
    i++;
  }
}
```

**Output:**
```
1
2
3
4
5
```

### Example 4: Break In Switch Case

This demonstrates break usage in switch statements, essential for preventing fall-through behavior.

```dart
void main() {
  var noOfMonth = 5;
  switch (noOfMonth) {
    case 1:
      print("Selected month is January.");
      break;
    case 2:
      print("Selected month is February.");
      break;
    case 3:
      print("Selected month is march.");
      break;
    case 4:
      print("Selected month is April.");
      break;
    case 5:
      print("Selected month is May.");
      break;
    case 6:
      print("Selected month is June.");
      break;
    case 7:
      print("Selected month is July.");
      break;
    case 8:
      print("Selected month is August.");
      break;
    case 9:
      print("Selected month is September.");
      break;
    case 10:
      print("Selected month is October.");
      break;
    case 11:
      print("Selected month is November.");
      break;
    case 12:
      print("Selected month is December.");
      break;
    default:
      print("Invalid month.");
      break;
  }
}
```

**Output:**
```
Selected month is May.
```

## Continue Statement

The continue statement skips the current iteration and proceeds to the next one, without terminating the loop.

**Syntax:**
```dart
continue;
```

### Example 1: Continue In Dart

This loop skips printing the number 5 but continues with subsequent iterations.

```dart
void main() {
  for (int i = 1; i <= 10; i++) {
    if (i == 5) {
      continue;
    }
    print(i);
  }
}
```

**Output:**
```
1
2
3
4
6
7
8
9
10
```

### Example 2: Continue In For Loop Dart

This descending loop skips the number 4 during iteration.

```dart
void main() {
  for (int i = 10; i >= 1; i--) {
    if (i == 4) {
      continue;
    }
    print(i);
  }
}
```

**Output:**
```
10
9
8
7
6
5
3
2
1
```

### Example 3: Continue In Dart While Loop

The continue statement skips iteration when the counter equals 5.

```dart
void main() {
  int i = 1;
  while (i <= 10) {
    if (i == 5) {
      i++;
      continue;
    }
    print(i);
    i++;
  }
}
```

**Output:**
```
1
2
3
4
6
7
8
9
10
```

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/break-and-continue-in-dart/
- **Fetched**: 2026-01-27
