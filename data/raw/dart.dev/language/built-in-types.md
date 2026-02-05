# Built-in Types in Dart

## Overview

Dart provides special support for several fundamental types. As stated on the documentation, "The Dart language has special support for the following: Numbers (int, double), Strings (String), Booleans (bool), Records, Functions, Lists, Sets, Maps, Runes, Symbols, and the value null (Null)."

Every variable in Dart references an object—an instance of a class—enabling you to use constructors for initialization.

## Numbers

Dart supports two numeric types:

**int**: Integer values up to 64 bits depending on platform. On native platforms: -2^63 to 2^63 - 1. On web: -2^53 to 2^53 - 1 (JavaScript numbers).

**double**: 64-bit floating-point numbers per IEEE 754 standard.

Both subtypes inherit from `num`, which provides operators (+, -, /, *) and methods like `abs()`, `ceil()`, and `floor()`.

### Examples

```dart
var x = 1;
var hex = 0xDEADBEEF;
var y = 1.1;
var exponents = 1.42e5;
```

Converting between strings and numbers:

```dart
var one = int.parse('1');
var onePointOne = double.parse('1.1');
String oneAsString = 1.toString();
String piAsString = 3.14159.toStringAsFixed(2);
```

Bitwise operations:

```dart
assert((3 << 1) == 6);
assert((3 | 4) == 7);
assert((3 & 4) == 0);
```

Digit separators improve readability:

```dart
var n1 = 1_000_000;
var n2 = 0.000_000_000_01;
var n3 = 0x00_14_22_01_23_45;
```

## Strings

A `String` holds a sequence of UTF-16 code units. Create strings with single or double quotes:

```dart
var s1 = 'Single quotes work well for string literals.';
var s2 = "Double quotes work just as well.";
var s3 = 'It\'s easy to escape the string delimiter.';
```

### String Interpolation

Use `${}` syntax to embed expressions:

```dart
var s = 'string interpolation';
assert('Dart has $s, which is very handy.' ==
    'Dart has string interpolation, which is very handy.');
assert('That deserves all caps. ${s.toUpperCase()} is very handy!' ==
    'That deserves all caps. STRING INTERPOLATION is very handy!');
```

### Concatenation

Concatenate via adjacent literals or the `+` operator:

```dart
var s1 = 'String ' 'concatenation' " works even over line breaks.";
var s2 = 'The + operator ' + 'works, as well.';
```

### Multi-line Strings

Triple quotes create multi-line strings:

```dart
var s1 = '''
You can create
multi-line strings like this one.
''';
var s2 = """This is also a
multi-line string.""";
```

### Raw Strings

Prefix with `r` to disable escape sequences:

```dart
var s = r'In a raw string, not even \n gets special treatment.';
```

## Booleans

Dart's `bool` type has exactly two instances: `true` and `false`. Type safety prevents implicit coercion:

```dart
var fullName = '';
assert(fullName.isEmpty);

var hitPoints = 0;
assert(hitPoints == 0);

var unicorn = null;
assert(unicorn == null);

var iMeantToDoThis = 0 / 0;
assert(iMeantToDoThis.isNaN);
```

## Runes and Grapheme Clusters

Runes expose Unicode code points. The `characters` package enables viewing user-perceived characters (grapheme clusters):

```dart
import 'package:characters/characters.dart';

void main() {
  var hi = 'Hi 🇩🇰';
  print(hi);
  print('The end of the string: ${hi.substring(hi.length - 1)}');
  print('The last character: ${hi.characters.last}');
}
```

Express Unicode using `\uXXXX` syntax (4 hex digits) or `\u{value}` for other lengths:

- Heart: `\u2665`
- Laughing emoji: `\u{1f606}`

## Symbols

A `Symbol` represents an operator or identifier in a Dart program. Useful for APIs referencing identifiers by name since minification changes names but not symbols.

Create symbols with `#` followed by the identifier:

```dart
#radix
#bar
```

Symbol literals are compile-time constants.

## Related Special Types

- **`Object`**: Superclass of all Dart classes except `Null`
- **`Enum`**: Superclass of all enums
- **`Future` and `Stream`**: Asynchronous programming
- **`Iterable`**: For-in loops and synchronous generators
- **`Never`**: Expression never finishes (e.g., always throws)
- **`dynamic`**: Disables static checking; prefer `Object` instead
- **`void`**: Value never used; common return type
