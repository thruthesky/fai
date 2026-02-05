# Where in Dart: Comprehensive Documentation

## Overview

The `where` method is a filtering mechanism in Dart that allows developers to extract specific items from collections. As stated in the tutorial, "It returns a new list containing all the elements that satisfy the condition."

## Method Signature

```dart
Iterable<E> where(
  bool test(
    E element
  )
)
```

The method accepts a test function that evaluates each element and returns a boolean value determining whether to include that element.

## Example 1: Filtering Odd Numbers

This demonstrates extracting odd numbers from a list of integers:

```dart
void main() {
  List<int> numbers = [2, 4, 6, 8, 10, 11, 12, 13, 14];

  List<int> oddNumbers = numbers.where((number) => number.isOdd).toList();
  print(oddNumbers);
}
```

**Output:** `[11, 13]`

The lambda expression `(number) => number.isOdd` tests each element against the odd property.

## Example 2: String-Based Filtering

This example shows filtering days beginning with specific letters:

```dart
void main() {
  List<String> days = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday"
  ];

  List<String> startWithS =
      days.where((element) => element.startsWith("S")).toList();

  print(startWithS);
}
```

**Output:** `[Sunday, Saturday]`

## Example 3: Filtering Map Collections

For map-based filtering, use `removeWhere()` to eliminate entries matching conditions:

```dart
void main() {
  Map<String, double> mathMarks = {
    "ram": 30,
    "mark": 32,
    "harry": 88,
    "raj": 69,
    "john": 15,
  };

  mathMarks.removeWhere((key, value) => value < 32);

  print(mathMarks);
}
```

**Output:** `{mark: 32.0, harry: 88.0, raj: 69.0}`

## Key Characteristics

- Compatible with **lists, sets, and maps**
- Returns an **iterable** that can be converted to a list via `.toList()`
- Enables **functional programming** patterns within Dart
- Supports **lambda expressions** for concise condition testing

---

## Source

- **URL**: https://dart-tutorial.com/collections/where-in-dart/
- **Fetched**: 2026-01-27
