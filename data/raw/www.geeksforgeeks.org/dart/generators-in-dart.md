# Generators in Dart

## Overview

Generators in Dart enable developers to produce sequences of values with ease. The language provides two distinct generator function types to accomplish this task.

## Types of Generators

### 1. Synchronous Generators

**Definition:** Synchronous generators return an `Iterable` object, which is a collection of values accessible in sequential order.

**Syntax:** Mark the function body with `sync*` and use `yield` statements to deliver values.

**Example:**
```dart
Iterable geeksForGeeks(int number) sync* {
    int geek = number;
    while (geek >= 0) {
        if (geek % 2 == 0) {
            yield geek;
        }
        geek--;
    }
}

void main() {
    print("------- Geeks For Geeks --------");
    print("Dart Synchronous Generator Example For Printing Even Numbers From 10 In Reverse Order:");
    geeksForGeeks(10).forEach(print);
}
```

**Output:**
```
------- Geeks For Geeks --------
Dart Synchronous Generator Example For Printing Even Numbers From 10 In Reverse Order:
10
8
6
4
2
0
```

### 2. Asynchronous Generators

**Definition:** Asynchronous generators return a `Stream` object, which provides a mechanism to receive a sequence of events. Events can be data elements or error notifications.

**Syntax:** Mark the function body with `async*` and use `yield` statements to deliver values.

**Example:**
```dart
Stream geeksForGeeks(int number) async* {
    int geek = 0;
    while (geek <= number) yield geek++;
}

void main() {
    print("-------- Geeks For Geeks -----------");
    print("Dart Asynchronous Generator Example For Printing Numbers Less Than 10:");
    geeksForGeeks(10).forEach(print);
}
```

**Output:**
```
-------- Geeks For Geeks -----------
Dart Asynchronous Generator Example For Printing Numbers Less Than 10:
0
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

## Key Differences

| Feature | Synchronous | Asynchronous |
|---------|-------------|--------------|
| Marker | `sync*` | `async*` |
| Return Type | `Iterable` | `Stream` |
| Use Case | Sequential data access | Event-based data handling |

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/generators-in-dart/
- **Fetched**: 2026-01-27
