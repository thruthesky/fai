# Strings in Dart - Complete Tutorial

## Introduction

A Dart string represents a sequence of UTF-16 code units. Following Python conventions, you may use either single or double quotes for string creation, with the datatype specified as `String` or `var`.

```dart
String string = "I love GeeksforGeeks";
var string1 = 'GeeksforGeeks is a great platform for upgrading skills';
```

Both declaration styles function identically in Dart environments.

## String Interpolation

Expressions can be embedded within strings using the `${expression}` syntax. For identifier-only expressions, the braces may be omitted.

```dart
void main() {
  var string = 'I do coding';
  var string1 = '$string on Geeks for Geeks';
  print(string1);
}
```

**Output:** `I do coding on Geeks for Geeks`

## String Concatenation

Dart supports multiple concatenation approaches:

### Using the + Operator

```dart
void main() {
  var str = 'Coding is ';
  var str1 = 'Fun';
  print(str + str1);
}
```

**Output:** `Coding is Fun`

### Implicit Concatenation

Adjacent string literals automatically concatenate:

```dart
var string = 'Geeks' 'for' 'Geeks';
print(string);
```

**Output:** `GeeksforGeeks`

## String Comparison

Use the `==` operator to check equality between strings. This compares each character systematically.

```dart
void main() {
  var str = 'Geeks';
  var str1 = 'Geeks';

  if (str == str1) {
    print('True');
  }
}
```

**Output:** `True`

## Raw Strings

Raw strings preserve special characters without escape sequence processing. Prefix the string with `r`.

```dart
void main() {
  var gfg = r'This is a raw string';
  print(gfg);
}
```

**Output:** `This is a raw string`

---

## String Properties

| Property | Description |
|----------|-------------|
| `length` | Returns character count |
| `isEmpty` | Returns `true` if empty |
| `isNotEmpty` | Returns `true` if non-empty |

### Implementation Example

```dart
void main() {
  var str = "GFG";
  print(str.length);        // 3
  print(str.isEmpty);       // false
  print(str.isNotEmpty);    // true
}
```

---

## String Methods

| Method | Purpose |
|--------|---------|
| `toLowerCase()` | Convert to lowercase |
| `toUpperCase()` | Convert to uppercase |
| `trim()` | Remove leading/trailing whitespace |
| `trimLeft()` | Remove leading whitespace |
| `trimRight()` | Remove trailing whitespace |
| `padLeft(width, [padding])` | Add left padding |
| `padRight(width, [padding])` | Add right padding |
| `contains(pattern)` | Check pattern existence |
| `startsWith(pattern, [index])` | Check starting substring |
| `endsWith(pattern)` | Check ending substring |
| `indexOf(pattern, [start])` | Find first occurrence index |
| `lastIndexOf(pattern, [start])` | Find last occurrence index |
| `replaceFirst(from, to, [start])` | Replace first match |
| `replaceAll(from, to)` | Replace all matches |
| `replaceRange(start, end, replacement)` | Replace character range |
| `split(pattern)` | Split into substring list |
| `substring(start, [end])` | Extract string portion |
| `codeUnitAt(index)` | Get Unicode code unit |
| `compareTo(other)` | Compare two strings |
| `toString()` | Return string representation |

### Comprehensive Method Example

```dart
void main() {
  var str = '  Dart Programming  ';

  print('Lowercase: ${str.toLowerCase()}');
  print('Uppercase: ${str.toUpperCase()}');
  print('Trimmed: "${str.trim()}"');
  print('Contains "Dart": ${str.contains("Dart")}');
  print('Starts with "  Dart": ${str.startsWith("  Dart")}');
  print('Index of "Dart": ${str.indexOf("Dart")}');
  print('Replace All " " with "-": ${str.replaceAll(" ", "-")}');
  print('Split by space: ${str.trim().split(" ")}');
}
```

---

## Key Takeaways

Dart strings provide flexible text manipulation through string interpolation, multiple concatenation techniques, and comprehensive built-in methods. The language supports raw strings for handling special characters, properties for checking string characteristics, and extensive methods for substring operations, case conversion, and pattern matching. Understanding these features enables efficient text processing in Dart applications.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/strings-in-dart/
- **Fetched**: 2026-01-27
