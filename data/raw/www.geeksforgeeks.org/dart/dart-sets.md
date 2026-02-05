# Dart Sets Tutorial

## Overview

Sets in Dart represent a specialized collection type where all elements must be unique. Sets in Dart is a special case in List, where all the inputs are unique i.e. it doesn't contain any repeated input. They function as unordered collections that automatically eliminate duplicates.

## Declaration

Two methods exist for declaring sets:

```dart
// Method 1
var variable_name = <variable_type>{};

// Method 2
Set<variable_type> variable_name = {};
```

### Example: Basic Declaration

```dart
void main() {
  var gfg1 = <String>{'GeeksForGeeks'};
  print("Output of first set: $gfg1");

  Set<String> gfg2 = {'GeeksForGeeks'};
  print("Output of second set: $gfg2");
}
```

**Output:**
```
Output of first set: {GeeksForGeeks}
Output of second set: {GeeksForGeeks}
```

## Unique Values Behavior

A key characteristic: duplicates are automatically ignored. When comparing a list with repeated values to a set with identical values, the set retains only unique entries.

```dart
void main() {
  var list = ['Geeks','For','Geeks'];
  print("Output of the list is: $list");

  var set = <String>{'Geeks','For','Geeks'};
  print("Output of the set is: $set");
}
```

**Output:**
```
Output of the list is: [Geeks, For, Geeks]
Output of the set is: {Geeks, For}
```

## Adding Elements

### Methods

- `.add(value)` - Adds single element
- `.addAll(iterable)` - Adds multiple elements

Duplicate values remain ignored even when explicitly added.

## Common Set Operations

| Operation | Syntax | Purpose |
|-----------|--------|---------|
| Access element | `variable_name.elementAt(index)` | Retrieve element at position |
| Get length | `variable_name.length` | Return set size |
| Check membership | `variable_name.contains(element)` | Boolean presence check |
| Remove element | `variable_name.remove(element)` | Delete specific element |
| Iterate | `variable_name.forEach(...)` | Process all elements |
| Clear all | `variable_name.clear()` | Remove all elements |

### Comprehensive Example

```dart
void main() {
  var gfg = <String>{'Hello Geek'};
  print("Value in the set is: $gfg");

  gfg.add("GeeksForGeeks");
  print("Values in the set is: $gfg");

  var geeks_name = {"Geek1","Geek2","Geek3"};
  gfg.addAll(geeks_name);
  print("Values in the set is: $gfg");

  var geek = gfg.elementAt(0);
  print("Element at index 0 is: $geek");

  int l = gfg.length;
  print("Length of the set is: $l");

  bool check = gfg.contains("GeeksForGeeks");
  print("The value of check is: $check");

  gfg.remove("Hello Geek");
  print("Values in the set is: $gfg");

  gfg.forEach((element) {
    if(element == "Geek1") {
      print("Found");
    } else {
      print("Not Found");
    }
  });

  gfg.clear();
  print("Values in the set is: $gfg");
}
```

## Type Conversion

### Set to List

```dart
List<String> list_variable = set_variable.toList();
```

The resulting list maintains uniqueness from the original set.

### Set to Map

```dart
var mapped = gfg.map((value) {
  return 'mapped $value';
});
```

## Set Operations

Dart supports three mathematical set operations:

### Union
Combines all elements from two sets:
```dart
var union = gfg1.union(gfg2);
```

### Intersection
Returns common elements:
```dart
var intersection = gfg1.intersection(gfg2);
```

### Difference
Returns elements in first set but not in second:
```dart
var difference = gfg2.difference(gfg1);
```

### Complete Example

```dart
void main() {
  var gfg1 = <String>{"GeeksForGeeks","Geek1","Geek2","Geek3"};
  print("Values in set 1 are:");
  print(gfg1);

  var gfg2 = <String>{"GeeksForGeeks","Geek3","Geek4","Geek5"};
  print("Values in set 2 are:");
  print(gfg2);

  print("Union of two sets is ${gfg1.union(gfg2)}");
  print("Intersection of two sets is ${gfg1.intersection(gfg2)}");
  print("Difference of two sets is ${gfg2.difference(gfg1)}");
}
```

## Conclusion

Sets provide an efficient mechanism for storing and managing unique values in Dart applications, with built-in operations supporting set algebra and type conversions for flexible data manipulation.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-sets/
- **Fetched**: 2026-01-27
