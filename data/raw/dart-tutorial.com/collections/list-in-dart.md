# Comprehensive Guide to Dart Lists

## Overview

Lists in Dart function similarly to arrays in other programming languages, allowing storage of multiple values in a single variable. They are denoted using square brackets `[]`.

## Creating Lists

Three common list creation patterns exist:

```dart
// Integer List
List<int> ages = [10, 30, 23];

// String List
List<String> names = ["Raj", "John", "Rocky"];

// Mixed List
var mixed = [10, "John", 18.8];
```

## List Types

### Fixed Length Lists

"The fixed-length Lists are defined with the specified length. You cannot change the size at runtime." Fixed-length lists are created using the `List.filled()` constructor.

```dart
void main() {
   var list = List<int>.filled(5, 0);
   print(list);
}
// Output: [0, 0, 0, 0, 0]
```

**Important:** Values can be modified, but new items cannot be added.

### Growable Lists

"A List defined without a specified length is called Growable List. The length of the growable List can be changed in runtime." This represents the most commonly used list type.

```dart
void main() {
   var list1 = [210, 21, 22, 33, 44, 55];
   print(list1);
}
```

## Accessing List Elements

### By Index

List indexing begins at zero. Access elements using bracket notation:

```dart
void main() {
  var list = [210, 21, 22, 33, 44, 55];
  print(list[0]); // 210
  print(list[2]); // 22
}
```

### By Value

Use `indexOf()` to locate an element's position:

```dart
void main() {
  var list = [210, 21, 22, 33, 44, 55];
  print(list.indexOf(22)); // 2
  print(list.indexOf(33)); // 3
}
```

## List Properties

| Property | Description |
|----------|-------------|
| `first` | Returns the initial element |
| `last` | Returns the final element |
| `isEmpty` | Returns `true` if empty |
| `isNotEmpty` | Returns `true` if contains elements |
| `length` | Returns total element count |
| `reversed` | Provides reverse-order iteration |
| `single` | Validates single-element lists and returns it |

### Accessing First and Last Elements

```dart
void main() {
   List<String> drinks = ["water", "juice", "milk", "coke"];
   print("First: ${drinks.first}");     // water
   print("Last: ${drinks.last}");       // coke
}
```

## Modifying Lists

### Changing Values

```dart
void main(){
   List<String> names = ["Raj", "John", "Rocky"];
   names[1] = "Bill";
   names[2] = "Elon";
   print(names);  // [Raj, Bill, Elon]
}
```

### Mutable vs Immutable

```dart
List<String> names = ["Raj", "John", "Rocky"]; // Mutable
names[1] = "Bill"; // Allowed

const List<String> names = ["Raj", "John", "Rocky"]; // Immutable
names[1] = "Bill"; // Not allowed
```

## Adding Elements

| Method | Function |
|--------|----------|
| `add()` | Appends single element at end |
| `addAll()` | Appends multiple elements |
| `insert()` | Inserts at specific position |
| `insertAll()` | Inserts multiple at position |

### Examples

```dart
// add()
var evenList = [2, 4, 6, 8, 10];
evenList.add(12);
// Result: [2, 4, 6, 8, 10, 12]

// addAll()
evenList.addAll([12, 14, 16, 18]);
// Result: [2, 4, 6, 8, 10, 12, 14, 16, 18]

// insert()
List myList = [3, 4, 2, 5];
myList.insert(2, 15);
// Result: [3, 4, 15, 2, 5]

// insertAll()
var myList = [3, 4, 2, 5];
myList.insertAll(1, [6, 7, 10, 9]);
// Result: [3, 6, 7, 10, 9, 4, 2, 5]
```

## Removing Elements

| Method | Function |
|--------|----------|
| `remove()` | Deletes specified value |
| `removeAt()` | Deletes at index position |
| `removeLast()` | Deletes final element |
| `removeRange()` | Deletes range of elements |

### Examples

```dart
// remove()
var list = [10, 20, 30, 40, 50];
list.remove(30);
// Result: [10, 20, 40, 50]

// removeAt()
list.removeAt(3);
// Result: [10, 11, 12, 14]

// removeLast()
list.removeLast();
// Result: [10, 20, 30, 40]

// removeRange()
list.removeRange(0, 3);
// Result: [40, 50]
```

## Replacing Elements

```dart
void main() {
  var list = [10, 15, 20, 25, 30];
  list.replaceRange(0, 4, [5, 6, 7, 8]);
  // Result: [5, 6, 7, 8, 30]
}
```

## Iterating Through Lists

### For Each Loop

```dart
void main() {
  List<int> list = [10, 20, 30, 40, 50];
  list.forEach((n) => print(n));
}
```

### Map Function

```dart
void main() {
  List<int> list = [10, 20, 30, 40, 50];
  var doubledList = list.map((n) => n * 2);
  print(doubledList);  // (20, 40, 60, 80, 100)
}
```

## Advanced List Operations

### Reversing Lists

```dart
void main() {
   List<String> drinks = ["water", "juice", "milk", "coke"];
   print("Reversed: ${drinks.reversed}");
   // Output: (coke, milk, juice, water)
}
```

### Combining Lists

```dart
void main() {
  List<String> names = ["Raj", "John", "Rocky"];
  List<String> names2 = ["Mike", "Subash", "Mark"];
  List<String> allNames = [...names, ...names2];
  print(allNames);
  // Result: [Raj, John, Rocky, Mike, Subash, Mark]
}
```

### Using Conditions

```dart
void main() {
  bool sad = false;
  var cart = ['milk', 'ghee', if (sad) 'Beer'];
  print(cart);  // [milk, ghee]
}
```

### Filtering with Where

```dart
void main(){
  List<int> numbers = [2, 4, 6, 8, 10, 11, 12, 13, 14];
  List<int> even = numbers.where((number) => number.isEven).toList();
  print(even);  // [2, 4, 6, 8, 10, 12, 14]
}
```

## Empty List Checking

```dart
void main() {
   List<String> drinks = ["water", "juice", "milk", "coke"];
   List<int> ages = [];

   print(drinks.isEmpty);      // false
   print(drinks.isNotEmpty);   // true
   print(ages.isEmpty);        // true
   print(ages.isNotEmpty);     // false
}
```

## Key Notes

- "Remember that List index starts with 0 and length always starts with 1."
- Choose lists when order matters and for frequent end appending
- Searching performance degrades with large list sizes

---

## Source

- **URL**: https://dart-tutorial.com/collections/list-in-dart/
- **Fetched**: 2026-01-27
