# Queues in Dart - Tutorial

## Introduction

A queue is a FIFO (First In First Out) data structure where the element that is added first will be deleted first. Queues manage ordered collections by accepting data at one end and removing it from the other, making them ideal for sequential data processing in Dart applications.

## Creating a Queue in Dart

### Using a Constructor

```dart
Queue variable_name = new Queue();
```

### From an Existing List

```dart
// With type notation (E)
Queue<E> variable_name = new Queue<E>.from(list_name);

// Without type notation
var variable_name = new Queue.from(list_name);
```

**Important:** Import the `dart:collection` module to use queues. Without it, you'll encounter compilation errors.

## Example 1: Basic Queue Creation and Operations

```dart
import 'dart:collection';

void main() {
  // Creating a Queue
  Queue<String> geek = new Queue<String>();

  // Printing default value
  print(geek);

  // Adding elements
  geek.add("Geeks");
  geek.add("For");
  geek.add("Geeks");

  // Printing inserted elements
  print(geek);
}
```

**Output:**
```
{}
{Geeks, For, Geeks}
```

## Example 2: Creating a Queue from a List

```dart
import 'dart:collection';

void main() {
  // Creating a List
  List<String> geek_list = ["Geeks","For","Geeks"];

  // Creating a Queue from the List
  Queue<String> geek_queue = new Queue<String>.from(geek_list);

  // Printing elements
  print(geek_queue);
}
```

**Output:**
```
{Geeks, For, Geeks}
```

## Queue Functions

| Function | Description |
|----------|-------------|
| `queue_name.add(element)` | Adds element to the back |
| `queue_name.addAll(collection)` | Adds all elements from a collection |
| `queue_name.addFirst(element)` | Adds element to the front |
| `queue_name.addLast(element)` | Adds element to the back |
| `queue_name.clear()` | Removes all elements |
| `queue_name.first()` | Returns first element |
| `queue_name.forEach(f(element))` | Iterates through all elements |
| `queue_name.isEmpty` | Checks if queue is empty |
| `queue_name.length` | Returns queue size |
| `queue_name.removeFirst()` | Removes first element |
| `queue_name.removeLast()` | Removes last element |

## Example 3: Comprehensive Queue Operations

```dart
import 'dart:collection';

void main() {
  Queue<String> geek = new Queue<String>();

  print(geek);  // {}

  geek.add("Geeks");
  print(geek);  // {Geeks}

  List<String> geek_data = ["For","Geeks"];
  geek.addAll(geek_data);
  print(geek);  // {Geeks, For, Geeks}

  geek.clear();
  print(geek);  // {}
  print(geek.isEmpty);  // true

  geek.addFirst("Geeks");
  print(geek);  // {Geeks}

  geek.addLast("For");
  geek.addLast("Geeks");
  print(geek);  // {Geeks, For, Geeks}

  print(geek.length);  // 3

  geek.removeFirst();
  print(geek);  // {For, Geeks}

  geek.removeLast();
  print(geek);  // {For}

  geek.forEach(print);  // For
}
```

## Key Takeaways

Queues in Dart follow FIFO principles and are essential for managing sequential data processing. They support flexible element manipulation through methods like `addFirst()`, `addLast()`, `removeFirst()`, and `removeLast()`. By mastering queue operations, developers can build efficient applications requiring ordered data handling and sequential task management.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/queues-in-dart/
- **Fetched**: 2026-01-27
