# Dart Data Types Tutorial

## Overview

In Dart, whenever a variable is created, each variable has an associated data type. Similar to C, C++, and Java, Dart provides specific types for representing and manipulating different kinds of values.

## Data Type Classification

| Data Type | Keyword | Description |
|-----------|---------|-------------|
| Number | int, double, num, BigInt | Numeric literals representation |
| Strings | String | Sequence of characters |
| Booleans | bool | True and false values |
| Lists | List | Ordered collection of elements |
| Sets | Set | Unordered collection of distinct elements |
| Maps | Map | Key-value pairs with unique keys |
| Runes | Runes | Unicode character manipulation |
| Symbols | Symbol | Identifier symbols for reflection |
| Null | Null | Absence of a value |

---

## 1. Numbers (int, double, num, BigInt)

Dart numbers hold numeric values and come in several varieties:

- **int**: Represents whole numbers (64-bit maximum)
- **double**: Represents 64-bit floating-point numbers with precision
- **num**: Inherited data type supporting both int and double
- **BigInt**: For very large integers exceeding `int` limits

### Declaring Integers

```dart
// Method 1: Explicit declaration
int age = 25;

// Method 2: Nullable declaration
int? count;

// Method 3: Using 'var' keyword
var year = 2024;
```

### Declaring Decimals

```dart
// Method 1: Explicit declaration
double pi = 3.1415;

// Method 2: Nullable declaration
double? percentage;

// Method 3: Using 'var' keyword
var temperature = 36.6;
```

### Using num for Both Types

```dart
num value = 10;
value = 10.5; // Allowed - num supports both
```

### Declaring Large Integers

```dart
BigInt bigNumber = BigInt.parse('987654321098765432109876543210');
```

### Example: Number Operations

```dart
void main() {
  int num1 = 2;
  double num2 = 1.5;

  print("$num1");
  print("$num2");

  var sum = num1 + num2;
  print("Sum = $sum");
}
```

**Output:**
```
2
1.5
Sum = 3.5
```

---

## 2. Strings

Strings represent a sequence of characters and are sequences of UTF-16 code units. Values use single or double quotation marks.

### Declaration

```dart
String str_name;
```

### Example: String Operations

```dart
void main() {
  String string = "Geeks for Geeks";

  String str = 'Coding is ';
  String str1 = 'Fun';

  print(string);
  print(str + str1);
}
```

**Output:**
```
Geeks for Geeks
Coding is Fun
```

---

## 3. Booleans

Boolean types represent true and false values using the `bool` keyword.

### Declaration

```dart
bool var_name;
```

### Example: Boolean Operations

```dart
void main() {
  bool val1 = true;

  String str = 'Coding is ';
  String str1 = 'Fun';

  bool val2 = (str == str1);

  print(val1);  // true
  print(val2);  // false
}
```

**Output:**
```
true
false
```

---

## 4. Lists

Lists represent an ordered list of elements similar to arrays in other languages.

### Variable Size Lists

```dart
// Empty growable list
List<int> var_name1 = [];

// Using List constructor
List<int> var_name3 = List.empty(growable: true);
```

### Fixed Size Lists

```dart
// Fixed-size list with default values
List<int> var_name1 = List<int>.filled(5, 0);

// Generated fixed-size list
List<int> var_name2 = List<int>.generate(5, (index) => index * 2);
```

### Example: List Operations

```dart
void main() {
  List<String> gfg = List<String>.filled(3, "default");

  gfg[0] = 'Geeks';
  gfg[1] = 'For';
  gfg[2] = 'Geeks';

  print(gfg);      // [Geeks, For, Geeks]
  print(gfg[0]);   // Geeks
}
```

**Output:**
```
[Geeks, For, Geeks]
Geeks
```

---

## 5. Sets

Sets store a list of distinct elements (unsorted) without duplicates.

### Declaration

```dart
// Using curly braces
Set<int> uniqueNumbers = {1, 2, 3, 3, 4};

// Using Set constructor
Set<String> cities = Set();
cities.add("New York");
```

### Example: Set Operations

```dart
void main() {
  Set<String> countries = {"USA", "India", "USA"};
  print(countries);  // {USA, India} - duplicates removed
}
```

**Output:**
```
{USA, India}
```

---

## 6. Maps

Maps represent collection of key-value pairs where keys are unique. They're dynamic collections useful for structured data.

### Empty Map Declaration

```dart
// Nullable declaration
Map? mapName;

// Type-safe empty map
Map<String, int> mapName4 = {};
```

### Map with Elements

```dart
// Using curly braces (preferred)
Map<String, String> myMap = {
  "First": "Geeks",
  "Second": "For",
  "Third": "Geeks",
};

// Using Map constructor
Map<String, int> mapExample = Map();
mapExample["One"] = 1;
mapExample["Two"] = 2;
```

### Example: Map Operations

```dart
void main() {
  Map<String, String> gfg = {};

  gfg['First'] = 'Geeks';
  gfg['Second'] = 'For';
  gfg['Third'] = 'Geeks';

  print(gfg);
}
```

**Output:**
```
{First: Geeks, Second: For, Third: Geeks}
```

---

## 7. Runes

Runes represent Unicode characters outside standard ASCII, particularly useful for emojis and special characters.

### Declaration

```dart
String heart = '\u2665';  // Heart symbol
```

### Example: Unicode Symbols

```dart
void main() {
  String heart = '\u2665';      // Heart
  String smiley = '\u263A';     // Smiley
  String star = '\u2605';       // Star
  String musicNote = '\u266B';  // Music Note

  print(heart);
  print(smiley);
  print(star);
  print(musicNote);
}
```

---

## 8. Symbols

Symbols are immutable identifiers primarily for reflection and dynamic programming.

### Declaration

```dart
Symbol sym1 = #mySymbol;
Symbol sym2 = Symbol("anotherSymbol");
```

### Example: Symbol Usage

```dart
void main() {
  Symbol sym1 = #dart;
  Symbol sym2 = #flutter;

  print(sym1);  // Symbol("dart")
  print(sym2);  // Symbol("flutter")

  Map<Symbol, String> symbolMap = {
    #language: "Dart",
    #framework: "Flutter",
  };

  print(symbolMap[#language]);   // Dart
  print(symbolMap[#framework]);  // Flutter
}
```

---

## 9. Null

Null indicates the lack of a value. Dart's null safety requires variables to be either nullable (?) or initialized.

### Declaration

```dart
String? name;  // Can be null
int? age;      // Can be null
```

### Example: Null Handling

```dart
void main() {
  String? name;
  int? age;

  name = "GFG";
  age = null;

  print(name ?? "Unknown");           // GFG
  print(age ?? "No age provided");    // No age provided

  int? length = name?.length;
  print(length);  // 3
}
```

---

## Key Notes

- If a variable's type isn't specified, it defaults to `dynamic`
- The `dynamic` keyword can be used as an explicit type annotation
- Dart supports null safety with nullable (?) type annotations
- Collections like Lists, Sets, and Maps use angle brackets for type specification: `List<int>`, `Map<String, String>`

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-data-types/
- **Fetched**: 2026-01-27
