# String in Dart - Complete Tutorial

## String in Dart

"String helps you to store text based data." You can represent names, addresses, or lengthy text. Strings contain letters, numbers, and special characters, using single, double, or triple quotes.

## String Declaration Examples

Single-line strings use single or double quotes; multi-line strings use triple quotes:

```dart
void main() {
   String text1 = 'This is an example of a single-line string.';
   String text2 = "This is an example of a single line string using double quotes.";
   String text3 = """This is a multiline line
string using the triple-quotes.
This is tutorial on dart strings.
""";
   print(text1);
   print(text2);
   print(text3);
}
```

## String Concatenation

Combine strings using the `+` operator or interpolation. Interpolation improves readability:

```dart
void main() {
String firstName = "John";
String lastName = "Doe";
print("Using +, Full Name is "+firstName + " " + lastName+".");
print("Using interpolation, full name is $firstName $lastName.");
}
```

## String Properties

- **codeUnits**: "Returns an unmodifiable list of the UTF-16 code units"
- **isEmpty**: Returns true for empty strings
- **isNotEmpty**: Returns false for empty strings
- **length**: "Returns the length of the string including space, tab, and newline characters"

## String Methods

### Case Conversion

```dart
void main() {
   String address1 = "Florida";
   String address2 = "TexAs";
   print("Address 1 in uppercase: ${address1.toUpperCase()}");
   print("Address 1 in lowercase: ${address1.toLowerCase()}");
}
```

### Trim Whitespace

```dart
void main() {
  String address1 = " USA";
  String address2 = "Japan  ";
  String address3 = "New Delhi";

  print("Result of address1 trim is ${address1.trim()}");
  print("Result of address2 trim is ${address2.trim()}");
  print("Result of address3 trim is ${address3.trim()}");
}
```

### Compare Strings

Returns 0 (equal), 1 (first greater), or -1 (first smaller):

```dart
void main() {
   String item1 = "Apple";
   String item2 = "Ant";
   String item3 = "Basket";

   print("Comparing item 1 with item 2: ${item1.compareTo(item2)}");
}
```

### Replace Substrings

```dart
void main() {
String text = "I am a good boy I like milk. Doctor says milk is good for health.";
String newText = text.replaceAll("milk", "water");
print("Original Text: $text");
print("Replaced Text: $newText");
}
```

### Split Strings

```dart
void main() {
  String allNames = "Ram, Hari, Shyam, Gopal";
  List<String> listNames = allNames.split(",");
  print("Value of listName is $listNames");
  print("List name at 0 index ${listNames[0]}");
}
```

### Convert to String

```dart
void main() {
int number = 20;
String result = number.toString();
print("Type of number is ${number.runtimeType}");
print("Type of result is ${result.runtimeType}");
}
```

### Substring Operations

```dart
void main() {
   String text = "I love computer";
   print("Print only computer: ${text.substring(7)}");
   print("Print only love: ${text.substring(2,6)}");
}
```

### Reverse a String

```dart
void main() {
  String input = "Hello";
  print("$input Reverse is ${input.split('').reversed.join()}");
}
```

### Capitalize First Letter

```dart
void main() {
  String text = "hello world";
  print("Capitalized first letter of String: ${text[0].toUpperCase()}${text.substring(1)}");
}
```

---

This tutorial covers essential string operations in Dart for text manipulation and processing tasks.

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/string-in-dart/
- **Fetched**: 2026-01-27
