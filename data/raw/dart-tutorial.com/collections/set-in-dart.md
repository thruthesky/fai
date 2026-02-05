# Comprehensive Dart Sets Documentation

## Overview

"Set is a unique collection of items. You cannot store duplicate values in the Set." It operates as an unordered collection, offering performance advantages when handling substantial datasets. Sets are ideal for storing distinct values without regard to insertion sequence, such as fruit names or day names. They are denoted using curly braces `{}`.

**Key distinction**: "The list allows you to add duplicate items, but the Set doesn't allow it."

## Syntax

```dart
Set <variable_type> variable_name = {};
```

## Creating Sets

To instantiate a Set, use the Set type annotation. `Set<String>` restricts entries to text values:

```dart
void main(){
  Set<String> fruits = {"Apple", "Orange", "Mango"};
  print(fruits);
}
```

Output:
```
{Apple, Orange, Mango}
```

## Set Properties

| Property | Function |
|----------|----------|
| `first` | Retrieves the initial value of the Set |
| `last` | Retrieves the final value of the Set |
| `isEmpty` | Returns boolean indicating emptiness |
| `isNotEmpty` | Returns boolean indicating non-emptiness |
| `length` | Returns the count of elements |

### Properties Example

```dart
void main() {
  Set<String> fruits = {"Apple", "Orange", "Mango", "Banana"};

  print("First Value is ${fruits.first}");
  print("Last Value is ${fruits.last}");
  print("Is fruits empty? ${fruits.isEmpty}");
  print("Is fruits not empty? ${fruits.isNotEmpty}");
  print("The length of fruits is ${fruits.length}");
}
```

Output:
```
First Value is Apple
Last Value is Banana
Is fruits empty? false
Is fruits not empty? true
The length of fruits is 4
```

## Checking Value Existence

The `contains()` method determines whether specific items exist within a Set, returning true or false:

```dart
void main(){
  Set<String> fruits = {"Apple", "Orange", "Mango"};
  print(fruits.contains("Mango"));
  print(fruits.contains("Lemon"));
}
```

Output:
```
true
false
```

## Adding and Removing Elements

| Method | Purpose |
|--------|---------|
| `add()` | Inserts a single element into the Set |
| `remove()` | Deletes a single element from the Set |

```dart
void main(){
  Set<String> fruits = {"Apple", "Orange", "Mango"};

  fruits.add("Lemon");
  fruits.add("Grape");

  print("After Adding Lemon and Grape: $fruits");

  fruits.remove("Apple");
  print("After Removing Apple: $fruits");
}
```

Output:
```
After Adding Lemon and Grape: {Apple, Orange, Mango, Lemon, Grape}
After Removing Apple: {Orange, Mango, Lemon, Grape}
```

## Adding Multiple Elements

The `addAll()` method enables insertion of multiple elements from a list into a Set:

```dart
void main(){
  Set<int> numbers = {10, 20, 30};
  numbers.addAll([40,50]);
  print("After adding 40 and 50: $numbers");
}
```

Output:
```
After adding 40 and 50: {10, 20, 30, 40, 50}
```

## Iterating Through Sets

Print all Set items using loop constructs:

```dart
void main(){
  Set<String> fruits = {"Apple", "Orange", "Mango"};

  for(String fruit in fruits){
    print(fruit);
  }
}
```

Output:
```
Apple
Orange
Mango
```

## Set Methods

| Method | Description |
|--------|-------------|
| `clear()` | Removes all elements from the Set |
| `difference()` | "Creates a new Set with the elements of this that are not in other" |
| `elementAt()` | Returns the element at a specified index position |
| `intersection()` | Identifies common elements between two sets |

### Clear Method

```dart
void main() {
  Set<String> fruits = {"Apple", "Orange", "Mango"};
  fruits.clear();
  print(fruits);
}
```

Output:
```
{}
```

### Difference Method

The difference operation generates a new Set containing elements present in the first set but absent from the second:

```dart
void main() {
  Set<String> fruits1 = {"Apple", "Orange", "Mango"};
  Set<String> fruits2 = {"Apple", "Grapes", "Banana"};

  final differenceSet = fruits1.difference(fruits2);
  print(differenceSet);
}
```

Output:
```
{Orange, Mango}
```

### Element At Method

Access Set values by index number (starting from 0):

```dart
void main() {
  Set<String> days = {"Sunday", "Monday", "Tuesday"};
  print(days.elementAt(2));
}
```

Output:
```
Tuesday
```

### Intersection Method

"The intersection method creates a new Set with the common elements in 2 Sets":

```dart
void main() {
  Set<String> fruits1 = {"Apple", "Orange", "Mango"};
  Set<String> fruits2 = {"Apple", "Grapes", "Banana"};

  final intersectionSet = fruits1.intersection(fruits2);
  print(intersectionSet);
}
```

Output:
```
{Apple}
```

---

## Source

- **URL**: https://dart-tutorial.com/collections/set-in-dart/
- **Fetched**: 2026-01-27
