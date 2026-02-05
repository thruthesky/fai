# Dart Programming - Map Tutorial

## Introduction

In Dart, maps function as dictionary-like data types that exist in key-value form (known as lock-key). These versatile collections allow flexible data storage without type restrictions, dynamically adjusting size as needed. Crucially, all keys must remain unique within any map instance.

## Declaration Methods

### Using Map Literals

Maps can be declared using literal syntax:

```dart
var map_name = { key1 : value1, key2 : value2, ..., key_n : value_n }
```

**Example 1: Basic Map Creation**

```dart
void main() {
  var gfg = {'position1' : 'Geek',
             'position2' : 'for',
             'position3' : 'Geeks'};

  print(gfg);
  print(gfg['position1']);  // Output: Geek
  print(gfg[0]);             // Output: null
}
```

**Output:**
```
{position1: Geek, position2: for, position3: Geeks}
Geek
null
```

**Example 2: String Concatenation**

```dart
void main() {
  var gfg = {'position1' : 'Geek' 'for' 'Geeks'};
  print(gfg);
}
```

**Output:**
```
{position1: GeekforGeeks}
```

Note: Adjacent strings automatically concatenate.

**Example 3: Inserting New Values**

```dart
void main() {
  var gfg = {'position1' : 'Geeks' 'for' 'Geeks'};
  print(gfg);

  gfg['position0'] = 'Welcome to ';
  print(gfg);
  print(gfg['position0'] + gfg['position1']);
}
```

**Output:**
```
{position1: GeeksforGeeks}
{position1: GeeksforGeeks, position0: Welcome to }
Welcome to GeeksforGeeks
```

### Using Map Constructors

Alternative declaration using constructors:

```dart
var map_name = new Map();
map_name[key] = value;
```

**Example 1: Constructor-Based Creation**

```dart
void main() {
  var gfg = new Map();

  gfg[0] = 'Geeks';
  gfg[1] = 'for';
  gfg[2] = 'Geeks';

  print(gfg);
  print(gfg[0]);
}
```

**Output:**
```
{0: Geeks, 1: for, 2: Geeks}
Geeks
```

**Example 2: Duplicate Key Behavior**

```dart
void main() {
  var gfg = new Map();

  gfg[0] = 'Geeks';
  gfg[0] = 'for';
  gfg[0] = 'Geeks';

  print(gfg);
  print(gfg[0]);
}
```

**Output:**
```
{0: Geeks}
Geeks
```

Note: Duplicate keys retain only the most recent assignment.

## Map Properties

| Property | Description |
|----------|-------------|
| length | Counts total elements in the map |
| isEmpty | Returns true if map has no elements |
| isNotEmpty | Returns true if map contains at least one element |
| keys | Returns an iterable of all keys |
| values | Returns an iterable of all values |
| entries | Returns an iterable of key-value pairs |

**Property Implementation:**

```dart
void main() {
  Map<String, int> scores = {'Alice': 95, 'Bob': 87, 'Charlie': 92};

  print("Map: $scores");
  print("Length: ${scores.length}");
  print("Is Empty: ${scores.isEmpty}");
  print("Is Not Empty: ${scores.isNotEmpty}");
  print("Keys: ${scores.keys}");
  print("Values: ${scores.values}");
}
```

## Map Methods

| Method | Description |
|--------|-------------|
| addAll() | Adds all key-value pairs from another map |
| remove(key) | Removes the entry for the specified key |
| clear() | Removes all entries from the map |
| containsKey(key) | Checks if the map contains the specified key |
| containsValue(value) | Checks if the map contains the specified value |
| forEach() | Executes a function for each key-value pair |
| update(key, update) | Updates the value for a key |
| putIfAbsent(key, ifAbsent) | Adds key-value if key doesn't exist |

**Method Implementation:**

```dart
void main() {
  Map<String, int> numbers = {'one': 1, 'two': 2};

  numbers.addAll({'three': 3, 'four': 4});
  print(numbers);

  numbers.remove('two');
  print(numbers);

  print(numbers.containsKey('one'));
  print(numbers.containsValue(3));

  numbers.forEach((key, value) {
    print('$key: $value');
  });

  numbers.update('one', (value) => value * 10);
  print(numbers);
}
```

## Conclusion

Maps represent fundamental Dart collection types. Maps operate as key-value pairs, allowing for flexible data storage where each key must be unique. Mastering these structures enables optimized data handling and improved application performance.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-programming-map/
- **Fetched**: 2026-01-27
