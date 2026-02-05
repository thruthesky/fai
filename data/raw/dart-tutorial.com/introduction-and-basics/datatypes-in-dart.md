# Data Types in Dart

## Overview

"Data types help you to categorize all the different types of data you use in your code." Dart supports eight built-in data types that enable developers to specify what values variables will contain.

## Built-In Types Reference

| Data Type | Keyword | Purpose |
|-----------|---------|---------|
| Numbers | int, double, num | Numeric values |
| Strings | String | Sequence of characters |
| Booleans | bool | True/false values |
| Lists | List | Ordered group of items |
| Maps | Map | Key-value pair collections |
| Sets | Set | Unordered unique values |
| Runes | runes | Unicode string values |
| Null | null | Null value representation |

## Numbers

Use `int` for whole numbers and `double` for decimals. The `num` type accommodates both:

```dart
void main() {
  int num1 = 100;
  double num2 = 130.2;
  num num3 = 50;
  num num4 = 50.4;

  num sum = num1 + num2 + num3 + num4;
  print("Sum is $sum");
}
```

### Rounding Decimals

Use `.toStringAsFixed(2)` to round to specified decimal places:

```dart
double price = 1130.2232323233233;
print(price.toStringAsFixed(2)); // Output: 1130.22
```

## Strings

Store text using single or double quotes:

```dart
String schoolName = "Diamond School";
String address = "New York 2140";
print("School name is $schoolName and address is $address");
```

### Multi-Line Strings

Use triple quotes for multi-line text:

```dart
String multiLineText = '''
This is Multi Line Text
with 3 single quote
I am also writing here.
''';
```

### Special Characters

| Character | Effect |
|-----------|--------|
| \n | New line |
| \t | Tab spacing |

### Raw Strings

Prefix with `r` to disable special character processing:

```dart
String withRawString = r"The value of price is \t $price";
// Output: The value of price is \t $price
```

## Type Conversion

### String to Int
```dart
String strvalue = "1";
int intvalue = int.parse(strvalue);
```

### String to Double
```dart
String strvalue = "1.1";
double doublevalue = double.parse(strvalue);
```

### Int to String
```dart
int one = 1;
String oneInString = one.toString();
```

### Double to Int
```dart
double num1 = 10.01;
int num2 = num1.toInt(); // Output: 10
```

## Booleans

Store true/false values for yes/no questions:

```dart
bool isMarried = true;
print("Married Status: $isMarried");
```

## Lists

"The list holds multiple values in a single variable." Access elements by zero-based index:

```dart
List<String> names = ["Raj", "John", "Max"];
print(names[0]); // Output: Raj
print(names.length); // Output: 3
```

## Sets

Unordered collections of unique items:

```dart
Set<String> weekday = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
print(weekday); // Duplicates automatically excluded
```

## Maps

Key-value pair objects:

```dart
Map<String, String> myDetails = {
  'name': 'John Doe',
  'address': 'USA',
  'fathername': 'Soe Doe'
};
print(myDetails['name']); // Output: John Doe
```

## Var Keyword

"Var automatically finds a data type" without explicit specification:

```dart
var name = "John Doe"; // Inferred as String
var age = 20; // Inferred as int
```

## Runes

Retrieve Unicode values from strings:

```dart
String value = "a";
print(value.runes); // Output: (97)
```

## Runtime Type Checking

Use `.runtimeType` to identify variable types:

```dart
var a = 10;
print(a.runtimeType); // Output: int
print(a is int); // Output: true
```

## Static vs. Dynamic Typing

Dart is an "optionally-typed language" supporting both approaches.

### Statically Typed
Types determined at compile time with error detection:

```dart
var myVariable = 50;
myVariable = "Hello"; // Error: incompatible assignment
```

### Dynamically Typed
Types determined at runtime allowing flexibility:

```dart
dynamic myVariable = 50;
myVariable = "Hello"; // Valid: Hello
```

**Recommendation**: "Using static type helps you to prevent writing silly mistakes in code."

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/datatypes-in-dart/
- **Fetched**: 2026-01-27
