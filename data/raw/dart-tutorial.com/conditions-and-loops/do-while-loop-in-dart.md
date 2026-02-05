# Do While Loop in Dart

## Overview

The do-while loop executes a code block repeatedly, with a key distinction: "the statement is executed before checking the condition." This guarantees at least one execution regardless of the condition's initial state.

## Syntax

```dart
do {
    statement1;
    statement2;
    // ...
    statementN;
} while(condition);
```

### How It Works

1. Statements execute first
2. Condition is evaluated afterward
3. If true, the loop repeats
4. If false, execution stops

## Example 1: Print 1 to 10

```dart
void main() {
  int i = 1;
  do {
    print(i);
    i++;
  } while (i <= 10);
}
```

**Output:**
```
1
2
3
4
5
6
7
8
9
10
```

## Example 2: Print 10 to 1

```dart
void main() {
  int i = 10;
  do {
    print(i);
    i--;
  } while (i >= 1);
}
```

**Output:** Numbers descending from 10 to 1

## Example 3: Sum of Natural Numbers

```dart
void main(){
  int total = 0;
  int n = 100;
  int i = 1;

  do {
    total = total + i;
    i++;
  } while(i <= n);

  print("Total is $total");
}
```

**Output:** `Total is 5050`

## When Condition Is False

Even with a false condition, the body executes once:

```dart
void main(){
  int number = 0;

  do {
    print("Hello");
    number--;
  } while(number > 1);
}
```

**Output:** `Hello` (prints exactly once despite the condition being false)

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/do-while-loop-in-dart/
- **Fetched**: 2026-01-27
