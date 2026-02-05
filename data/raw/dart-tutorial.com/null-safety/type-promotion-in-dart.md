# Type Promotion in Dart

## Definition

"Type promotion in dart means that dart automatically converts a value of one type to another type" when the compiler identifies a specific type.

## How Type Promotion Works

Type promotion occurs in two main scenarios:

1. Converting from general types to specific subtypes
2. Converting from nullable types to non-nullable types

## Example 1: General to Specific Subtypes

When a variable declared as `Object` is checked with type guards, it can be promoted to a more specific type:

```dart
void main(){
  Object name = "Pratik";

  if(name is String) {
    // name promoted from Object to String
    print("The length of name is ${name.length}");
  }
}
```

**Output:** `The length of name is 6`

## Example 2: Non-Nullable String Promotion

A variable assigned values in all conditional branches gets promoted to a non-nullable type:

```dart
void main(){
  String result;
  if(DateTime.now().hour < 12) {
    result = "Good Morning";
  } else {
    result = "Good Afternoon";
  }
  print("Result is $result");
  print("Length of result is ${result.length}");
}
```

**Output:** `Result is Good Afternoon` / `Length of result is 15`

## Example 3: Null Exception Handling

Methods can throw exceptions when nullable parameters are null:

```dart
void printLength(String? text){
  if(text == null) {
    throw Exception("The text is null");
  }
  print("Length of text is ${text.length}");
}

void main() {
  printLength("Hello");
}
```

**Output:** `Length of text is 5`

## Example 4: Nullable to Non-Nullable with Type Check

Using type checking to promote nullable values:

```dart
import 'dart:math';

class DataProvider{
  String? get stringorNull => Random().nextBool() ? "Hello" : null;

  void myMethod(){
    String? value = stringorNull;
    if(value is String){
      print("The length of value is ${value.length}");
    } else {
      print("The value is not string.");
    }
  }
}

void main() {
  DataProvider().myMethod();
}
```

**Output:** Either `The length of value is 5` or `The value is not string.` (random)

---

## Source

- **URL**: https://dart-tutorial.com/null-safety/type-promotion-in-dart/
- **Fetched**: 2026-01-27
