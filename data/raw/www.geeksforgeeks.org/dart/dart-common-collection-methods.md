# Dart - Common Collection Methods Tutorial

## Overview

List, Set, and Map share common functionalities found in many collections. The Iterable class defines some of this common functionality, implemented by both List and Set.

## 1. isEmpty() or isNotEmpty

Check whether a list, set, or map contains items using these methods.

**Example:**

```dart
void main() {
  // Declaring an empty list of coffees
  var coffees = [];

  // Declaring a list of teas with some values
  var teas = ['green', 'black', 'chamomile', 'earl grey'];

  // Checking if the 'coffees' list is empty
  print(coffees.isEmpty); // Output: true

  // Checking if the 'teas' list is not empty
  print(teas.isNotEmpty); // Output: true
}
```

**Output:**
```
true
true
```

## 2. forEach()

Apply a function to each item in a collection by using forEach().

**Example:**

```dart
void main() {
  // Declaring a list of tea types
  var teas = ['green', 'black', 'chamomile', 'earl grey'];

  // Using map() to convert each tea name to uppercase
  var loudTeas = teas.map((tea) => tea.toUpperCase());

  // Iterating and printing each tea name
  loudTeas.forEach(print);
}
```

**Output:**
```
GREEN
BLACK
CHAMOMILE
EARL GREY
```

## 3. where(), any(), and every()

- **where()**: Retrieves items matching a condition
- **any()**: Checks if at least one item matches a condition
- **every()**: Checks if all items match a condition

**Example:**

```dart
void main() {
  var teas = ['green', 'black', 'chamomile', 'earl grey'];

  // Function to check if a tea is decaffeinated
  bool isDecaffeinated(String teaName) => teaName == 'chamomile';

  // Using any() - checks if at least one tea matches
  print(teas.any(isDecaffeinated)); // Output: true

  // Using every() - checks if all teas match
  print(!teas.every(isDecaffeinated)); // Output: true
}
```

**Output:**
```
true
true
```

## Conclusion

Mastering iterable methods like isEmpty(), forEach(), where(), any(), and every() enables developers to write cleaner and more efficient Dart code while working with collections effectively.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-common-collection-methods/
- **Fetched**: 2026-01-27
