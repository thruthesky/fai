# Dart Generics Guide

## Overview

In Dart, collections are heterogeneous by default. Generics enable collections to store homogeneous values, ensuring type safety. These type-safe collections enforce a single required data type for all elements.

**Syntax:**
```
Collection_name <data_type> identifier = new Collection_name<data_type>()
```

Dart supports generics for List, Set, Map, and Queue collections.

---

## Generic List

A List represents an ordered collection of objects, functioning as an array implementation.

**Example:**
```dart
main() {
  List<int> listEx = [];
  listEx.add(341);
  listEx.add(1);
  listEx.add(23);

  for (int element in listEx) {
    print(element);
  }
}
```

**Output:**
```
341
1
23
```

---

## Generic Set

A Set is a collection where each object can exist only once, preventing duplicate entries.

**Example:**
```dart
main() {
  Set<int> SetEx = new Set<int>();
  SetEx.add(12);
  SetEx.add(3);
  SetEx.add(4);
  SetEx.add(3);  // Duplicate - won't be added

  for (int element in SetEx) {
    print(element);
  }
}
```

**Output:**
```
12
3
4
```

---

## Generic Map

A Map is a dynamic collection storing key-value pairs with specified types.

**Example:**
```dart
main() {
  Map<String, int> mp = {
    'Ankur': 1,
    'Arnav': 002,
    'Shivam': 003
  };
  print('Map :${mp}');
}
```

**Output:**
```
Map :{Ankur: 1, Arnav: 2, Shivam: 3}
```

---

## Generic Queue

A Queue implements FIFO (First In First Out) insertion. Dart queues support manipulation at both ends.

**Example:**
```dart
import 'dart:collection';

main() {
  Queue<int> q = new Queue<int>();

  q.addLast(1);    // Insert at end
  q.addLast(2);    // Insert at end
  q.addFirst(3);   // Insert at start
  q.addLast(4);    // Insert at end

  // Current order: 3 1 2 4
  q.removeFirst(); // Remove first element (3)

  for(int element in q) {
    print(element);
  }
}
```

**Output:**
```
1
2
4
```

---

## Key Benefits

- **Type Safety**: Enforces homogeneous data types within collections
- **Error Prevention**: Compile-time type checking catches errors early
- **Code Clarity**: Explicit type declarations improve readability

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-generics/
- **Fetched**: 2026-01-27
