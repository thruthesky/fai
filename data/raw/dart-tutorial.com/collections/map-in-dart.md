# Dart Maps: Comprehensive Documentation

## Overview

"In a Map, data is stored as keys and values. In Map, each key must be unique." Maps function similarly to HashMaps and Dictionaries found in other programming languages.

## Creating Maps

Maps are declared with type parameters specifying the key and value types:

```dart
void main(){
  Map<String, String> countryCapital = {
    'USA': 'Washington, D.C.',
    'India': 'New Delhi',
    'China': 'Beijing'
  };
  print(countryCapital);
}
```

**Output:**
```
{USA: Washington, D.C., India: New Delhi, China: Beijing}
```

## Accessing Values

Retrieve values by referencing their associated keys using bracket notation:

```dart
void main(){
  Map<String, String> countryCapital = {
    'USA': 'Washington, D.C.',
    'India': 'New Delhi',
    'China': 'Beijing'
  };
  print(countryCapital["USA"]);
}
```

**Output:** `Washington, D.C.`

## Map Properties

| Property | Purpose |
|----------|---------|
| `keys` | Retrieves all keys |
| `values` | Retrieves all values |
| `isEmpty` | Returns boolean indicating empty status |
| `isNotEmpty` | Returns boolean indicating non-empty status |
| `length` | Returns the number of key-value pairs |

### Property Example

```dart
void main() {
  Map<String, double> expenses = {
    'sun': 3000.0,
    'mon': 3000.0,
    'tue': 3234.0,
  };

  print("All keys of Map: ${expenses.keys}");
  print("All values of Map: ${expenses.values}");
  print("Is Map empty: ${expenses.isEmpty}");
  print("Is Map not empty: ${expenses.isNotEmpty}");
  print("Length of map is: ${expenses.length}");
}
```

**Output:**
```
All keys of Map: (sun, mon, tue)
All values of Map: (3000, 3000, 3234)
Is Map empty: false
Is Map not empty: true
Length of map is: 3
```

## Adding Elements

Introduce new key-value pairs using assignment syntax:

```dart
void main(){
  Map<String, String> countryCapital = {
    'USA': 'Washington, D.C.',
    'India': 'New Delhi',
    'China': 'Beijing'
  };
  countryCapital['Japan'] = 'Tokio';
  print(countryCapital);
}
```

**Output:**
```
{USA: Washington, D.C., India: New Delhi, China: Beijing, Japan: Tokio}
```

## Updating Elements

Modify existing values by reassigning through their keys:

```dart
void main(){
  Map<String, String> countryCapital = {
    'USA': 'Nothing',
    'India': 'New Delhi',
    'China': 'Beijing'
  };
  countryCapital['USA'] = 'Washington, D.C.';
  print(countryCapital);
}
```

**Output:**
```
{USA: Washington, D.C., India: New Delhi, China: Beijing}
```

## Map Methods

| Method | Function |
|--------|----------|
| `keys.toList()` | Converts all keys to a List |
| `values.toList()` | Converts all values to a List |
| `containsKey('key')` | Checks key existence |
| `containsValue('value')` | Checks value existence |
| `clear()` | Removes all elements |
| `removeWhere()` | Conditionally removes elements |

### Converting Keys and Values to Lists

```dart
void main() {
  Map<String, double> expenses = {
    'sun': 3000.0,
    'mon': 3000.0,
    'tue': 3234.0,
  };

  print("All keys of Map: ${expenses.keys}");
  print("All values of Map: ${expenses.values}");
  print("All keys with List: ${expenses.keys.toList()}");
  print("All values with List: ${expenses.values.toList()}");
}
```

**Output:**
```
All keys of Map: (sun, mon, tue)
All values of Map: (3000, 3000, 3234)
All keys with List: [sun, mon, tue]
All values with List: [3000, 3000, 3234]
```

## Checking for Keys and Values

```dart
void main() {
  Map<String, double> expenses = {
    'sun': 3000.0,
    'mon': 3000.0,
    'tue': 3234.0,
  };

  print("Contains key sun: ${expenses.containsKey("sun")}");
  print("Contains key abc: ${expenses.containsKey("abc")}");
  print("Contains value 3000.0: ${expenses.containsValue(3000.0)}");
  print("Contains value 100.0: ${expenses.containsValue(100.0)}");
}
```

**Output:**
```
Contains key sun: true
Contains key abc: false
Contains value 3000.0: true
Contains value 100.0: false
```

## Removing Elements

Delete items using the `remove()` method:

```dart
void main(){
  Map<String, String> countryCapital = {
    'USA': 'Nothing',
    'India': 'New Delhi',
    'China': 'Beijing'
  };

  countryCapital.remove("USA");
  print(countryCapital);
}
```

**Output:**
```
{India: New Delhi, China: Beijing}
```

## Iterating Over Maps

### Using For-In Loop with Entries

```dart
void main(){
  Map<String, dynamic> book = {
    'title': 'Misson Mangal',
    'author': 'Kuber Singh',
    'page': 233
  };

  for(MapEntry book in book.entries){
    print('Key is ${book.key}, value ${book.value}');
  }
}
```

**Output:**
```
Key is title, value Misson Mangal
Key is author, value Kuber Singh
Key is page, value 233
```

### Using forEach Method

```dart
void main(){
  Map<String, dynamic> book = {
    'title': 'Misson Mangal',
    'author': 'Kuber Singh',
    'page': 233
  };

  book.forEach((key,value)=>
    print('Key is $key and value is $value'));
}
```

**Output:**
```
Key is title and value is Misson Mangal
Key is author and value is Kuber Singh
Key is page and value is 233
```

## Conditional Removal with removeWhere

Filter maps based on specific conditions:

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

**Output:**
```
{mark: 32.0, harry: 88.0, raj: 69.0}
```

This example demonstrates retaining only entries where values meet or exceed a threshold of 32.

---

## Source

- **URL**: https://dart-tutorial.com/collections/map-in-dart/
- **Fetched**: 2026-01-27
