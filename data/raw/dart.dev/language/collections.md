# Dart Collections - Complete Guide

## Overview

Dart offers built-in support for three primary collection types: lists, sets, and maps. Understanding these structures is essential for effective Dart programming.

## Lists

Lists represent "ordered groups of objects" and are the most commonly used collection type. They use zero-based indexing.

### Basic List Creation

```dart
var list = [1, 2, 3];
```

Key characteristics:
- Type inference: `List<int>` is automatically inferred
- Trailing commas are permitted without affecting the collection
- Access elements using bracket notation: `list[0]`
- Retrieve length with `.length` property

### List Operations

```dart
var list = [1, 2, 3];
assert(list.length == 3);
assert(list[1] == 2);

list[1] = 1;
assert(list[1] == 1);
```

### Constant Lists

```dart
var constantList = const [1, 2, 3];
// constantList[1] = 1; // Causes error - immutable
```

## Sets

Sets are "unordered collection[s] of unique elements," preventing duplicates through their design.

### Set Creation

```dart
var halogens = {'fluorine', 'chlorine', 'bromine', 'iodine', 'astatine'};
```

Type inference assigns `Set<String>`.

### Empty Sets

```dart
var names = <String>{};
// Set<String> names = {}; // Alternative syntax
// var names = {}; // Creates Map, not Set
```

**Important distinction**: Empty braces `{}` default to `Map` type without explicit type annotation.

### Set Operations

```dart
var elements = <String>{};
elements.add('fluorine');
elements.addAll(halogens);
assert(elements.length == 5);
```

### Constant Sets

```dart
final constantSet = const {
  'fluorine',
  'chlorine',
  'bromine',
  'iodine',
  'astatine',
};
```

## Maps

Maps store "key-value pair[s]" where each key maps to exactly one value, though multiple keys can reference the same value.

### Map Creation

```dart
var gifts = {
  'first': 'partridge',
  'second': 'turtledoves',
  'fifth': 'golden rings',
};

var nobleGases = {2: 'helium', 10: 'neon', 18: 'argon'};
```

Type inference produces `Map<String, String>` and `Map<int, String>` respectively.

### Constructor Syntax

```dart
var gifts = Map<String, String>();
gifts['first'] = 'partridge';
gifts['second'] = 'turtledoves';
gifts['fifth'] = 'golden rings';
```

### Map Operations

```dart
var gifts = {'first': 'partridge'};
gifts['fourth'] = 'calling birds'; // Add pair

assert(gifts['first'] == 'partridge'); // Retrieve value

assert(gifts['fifth'] == null); // Missing key returns null

assert(gifts.length == 2); // Count pairs
```

### Constant Maps

```dart
final constantMap = const {2: 'helium', 10: 'neon', 18: 'argon'};
```

## Collection Elements

Collection literals support various element types for flexible data construction.

### Expression Elements

Simple expressions that evaluate and insert values:

```dart
<expression>
```

### Map Entry Elements

Key-value pairs within maps:

```dart
<key_expression>: <value_expression>
```

### Null-Aware Elements

Conditionally insert values only when non-null (requires language version 3.8+):

```dart
?<expression>
```

```dart
int? absentValue = null;
int? presentValue = 3;
var items = [
  1,
  ?absentValue,
  ?presentValue,
  absentValue,
  5,
]; // [1, 3, null, 5]
```

Map entry examples:

```dart
String? presentKey = 'Apple';
int? presentValue = 3;
int? absentValue = null;

var itemsA = {presentKey: absentValue}; // {Apple: null}
var itemsB = {presentKey: ?absentValue}; // {}
```

### Spread Elements

Iterate over sequences and insert all values:

```dart
...<sequence_expression>
```

```dart
var a = [1, 2, null, 4];
var items = [0, ...a, 5]; // [0, 1, 2, null, 4, 5]
```

### Null-Aware Spread Elements

Safely spread nullable collections:

```dart
...?<sequence_expression>
```

```dart
List<int>? a = null;
var b = [1, null, 3];
var items = [0, ...?a, ...?b, 4]; // [0, 1, null, 3, 4]
```

**Error example** (spread on nullable without null-awareness):

```dart
List<String> buildCommandLine(
  String executable,
  List<String> options, [
  List<String>? extraOptions,
]) {
  return [
    executable,
    ...options,
    ...extraOptions, // Compile-time error
  ];
}
```

**Corrected version**:

```dart
List<String> buildCommandLine(
  String executable,
  List<String> options, [
  List<String>? extraOptions,
]) {
  return [
    executable,
    ...options,
    ...?extraOptions, // OK - null-aware
  ];
}
```

### If Elements

Conditionally include elements using boolean expressions or pattern matching:

```dart
if (<bool_expression>) <result>
if (<expression> case <pattern>) <result>
if (<bool_expression>) <result> else <result>
```

Boolean examples:

```dart
var includeItem = true;
var items = [0, if (includeItem) 1, 2, 3]; // [0, 1, 2, 3]

var items = [0, if (!includeItem) 1, 2, 3]; // [0, 2, 3]

var name = 'apple';
var items = [0, if (name == 'orange') 1 else 10, 2, 3]; // [0, 10, 2, 3]
```

Pattern matching examples:

```dart
Object data = 123;
var typeInfo = [
  if (data case int i) 'Data is an integer: $i',
  if (data case String s) 'Data is a string: $s',
];
```

### For Elements

Iterate within collection literals:

```dart
for (<expression> in <collection>) <result>
for (<init>; <condition>; <increment>) <result>
```

```dart
var numbers = [2, 3, 4];
var items = [1, for (var n in numbers) n * n, 7]; // [1, 4, 9, 16, 7]

var items = [1, for (var x = 5; x > 2; x--) x, 7]; // [1, 5, 4, 3, 7]
```

### Nested Control Flow Elements

Combine multiple control flow structures:

```dart
var numbers = [1, 2, 3, 4, 5, 6, 7];
var items = [
  0,
  for (var n in numbers)
    if (n.isEven) n,
  8,
]; // [0, 2, 4, 6, 8]
```

Complex nesting example:

```dart
var nestItems = true;
var ys = [1, 2, 3, 4];
var items = [
  if (nestItems) ...[
    for (var x = 0; x < 3; x++)
      for (var y in ys)
        if (x < y) x + y * 10,
  ],
]; // [10, 20, 30, 40, 21, 31, 41, 32, 42]
```

## Key Takeaways

- Lists provide ordered, indexed access to elements
- Sets enforce uniqueness automatically
- Maps associate keys with values efficiently
- Collection elements support sophisticated control flow directly in literals
- Null-aware operators prevent runtime errors when handling nullable types
- Nested control structures enable powerful, functional-style data transformations
