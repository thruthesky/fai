# Dart Loop Control Statements (Break and Continue)

## Overview

Dart provides two essential loop control mechanisms: **break** and **continue** statements that manage loop execution flow based on specific conditions.

## Break Statement

### Purpose

This statement is used to break the flow of control of the loop i.e. if it is used within a loop then it will terminate the loop whenever encountered.

### Syntax

```dart
break;
```

### Using Break in While Loop

```dart
void main() {
    int count = 1;

    while (count <= 10) {
        print("Geek, you are inside loop $count");
        count++;

        if (count == 4) {
            break;
        }
    }
    print("Geek, you are out of while loop");
}
```

**Output:**
```
Geek, you are inside loop 1
Geek, you are inside loop 2
Geek, you are inside loop 3
Geek, you are out of while loop
```

### Using Break in Do-While Loop

```dart
void main() {
    int count = 1;

    do {
        print("Geek, you are inside loop $count");
        count++;

        if (count == 5) {
            break;
        }
    } while (count <= 10);

    print("Geek, you are out of do..while loop");
}
```

### Using Break in For Loop

```dart
void main() {
    for (int i = 1; i <= 10; ++i) {
        if (i == 2)
            break;

        print("Geek, you are inside loop $i");
    }

    print("Geek, you are out of loop");
}
```

**Output:**
```
Geek, you are inside loop 1
Geek, you are out of loop
```

---

## Continue Statement

### Purpose

When a continue statement is encountered in a loop it doesn't terminate the loop but rather jump the flow to next iteration.

### Syntax

```dart
continue;
```

### Using Continue in While Loop

```dart
void main() {
    int count = 0;

    while (count <= 10) {
        count++;

        if (count == 4) {
            print("Number 4 is skipped");
            continue;
        }

        print("Geek, you are inside loop $count");
    }

    print("Geek, you are out of while loop");
}
```

**Output:**
```
Geek, you are inside loop 1
Geek, you are inside loop 2
Geek, you are inside loop 3
Number 4 is skipped
Geek, you are inside loop 5
...
Geek, you are out of while loop
```

### Using Continue in Do-While Loop

Similar behavior applies to do-while loops, skipping the specified iteration while maintaining loop continuity.

### Using Continue in For Loop

```dart
void main() {
    for (int i = 1; i <= 10; ++i) {
        if (i == 2) {
            print("Geek, you are inside loop $i");
            continue;
        }
    }

    print("Geek, you are out of loop");
}
```

---

## Key Differences Summary

| Feature | Break | Continue |
|---------|-------|----------|
| **Action** | Exits loop entirely | Skips current iteration |
| **Loop Termination** | Yes | No |
| **Use Case** | Stop when condition met | Bypass specific iterations |
| **Compatible With** | for, while, do-while | for, while, do-while |

---

## Conclusion

Loop control statements optimize iteration efficiency. The break statement halts execution when criteria are satisfied, while continue permits selective iteration skipping without terminating the loop structure. Both enhance code control and performance.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-loop-control-statements-break-and-continue/
- **Fetched**: 2026-01-27
