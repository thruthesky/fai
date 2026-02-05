# dart:core Library Documentation

## Overview

The `dart:core` library provides essential built-in functionality automatically imported into every Dart program. The [API reference](https://api.dart.dev/dart-core/dart-core-library.html) documents all available classes and methods.

## Printing to Console

Use the top-level `print()` function to output any object's string representation:

```dart
print(anObject);
print('I drink $tea.');
```

## Numbers

The library defines `num`, `int`, and `double` classes for numeric operations.

### Parsing Numbers

Convert strings to numbers using `parse()` methods:

```dart
assert(int.parse('42') == 42);
assert(int.parse('0x42') == 66);
assert(double.parse('0.50') == 0.5);
```

Use `num.parse()` for automatic type detection:

```dart
assert(num.parse('42') is int);
assert(num.parse('0.50') is double);
```

Specify base with the `radix` parameter:

```dart
assert(int.parse('42', radix: 16) == 66);
```

### Converting Numbers to Strings

```dart
assert(42.toString() == '42');
assert(123.456.toString() == '123.456');
assert(123.456.toStringAsFixed(2) == '123.46');
assert(123.456.toStringAsPrecision(2) == '1.2e+2');
```

## Strings and Regular Expressions

Strings are immutable UTF-16 code unit sequences.

### Searching Strings

```dart
assert('Never odd or even'.contains('odd'));
assert('Never odd or even'.startsWith('Never'));
assert('Never odd or even'.endsWith('even'));
assert('Never odd or even'.indexOf('odd') == 6);
```

### Extracting Data

```dart
assert('Never odd or even'.substring(6, 9) == 'odd');
var parts = 'progressive web apps'.split(' ');
assert(parts[0] == 'progressive');
assert('Never odd or even'[0] == 'N');

var codeUnitList = 'Never odd or even'.codeUnits.toList();
assert(codeUnitList[0] == 78);
```

For grapheme clusters (user-perceived characters), use the [`characters` package](https://pub.dev/packages/characters).

### Case Conversion

```dart
assert('web apps'.toUpperCase() == 'WEB APPS');
assert('WEB APPS'.toLowerCase() == 'web apps');
```

### Trimming and Checking Emptiness

```dart
assert('  hello  '.trim() == 'hello');
assert(''.isEmpty);
assert('  '.isNotEmpty);
```

### Replacing Content

Strings are immutable, so `replaceAll()` returns a new String:

```dart
var greetingTemplate = 'Hello, NAME!';
var greeting = greetingTemplate.replaceAll(RegExp('NAME'), 'Bob');
assert(greeting != greetingTemplate);
```

### Building Strings

Use `StringBuffer` for efficient string construction:

```dart
var sb = StringBuffer();
sb
  ..write('Use a StringBuffer for ')
  ..writeAll(['efficient', 'string', 'creation'], ' ')
  ..write('.');

var fullString = sb.toString();
assert(fullString == 'Use a StringBuffer for efficient string creation.');
```

### Regular Expressions

The `RegExp` class supports pattern matching:

```dart
var digitSequence = RegExp(r'\d+');
var someDigits = 'llamas live 15 to 20 years';

assert(someDigits.contains(digitSequence));
var exedOut = someDigits.replaceAll(digitSequence, 'XX');
assert(exedOut == 'llamas live XX to XX years');
```

Work directly with matches:

```dart
assert(digitSequence.hasMatch(someDigits));
for (final match in digitSequence.allMatches(someDigits)) {
  print(match.group(0)); // 15, then 20
}
```

## Collections

### Lists

Create and manipulate lists:

```dart
var grains = <String>[];
var fruits = ['apples', 'oranges'];
fruits.add('kiwis');
fruits.addAll(['grapes', 'bananas']);
assert(fruits.length == 5);

var appleIndex = fruits.indexOf('apples');
fruits.removeAt(appleIndex);
fruits.clear();

var vegetables = List.filled(99, 'broccoli');
```

Access by index and find elements:

```dart
var fruits = ['apples', 'oranges'];
assert(fruits[0] == 'apples');
assert(fruits.indexOf('apples') == 0);
```

Sort using comparison functions:

```dart
var fruits = ['bananas', 'apples', 'oranges'];
fruits.sort((a, b) => a.compareTo(b));
assert(fruits[0] == 'apples');
```

Use parameterized types for type safety:

```dart
var fruits = <String>[];
fruits.add('apples');
var fruit = fruits[0];
assert(fruit is String);
```

See the [List API reference](https://api.dart.dev/dart-core/List-class.html) for complete method documentation.

### Sets

Sets store unique, unordered items:

```dart
var ingredients = <String>{};
ingredients.addAll(['gold', 'titanium', 'xenon']);
assert(ingredients.length == 3);

ingredients.add('gold'); // No effect, already exists
ingredients.remove('gold');

var atomicNumbers = Set.from([79, 22, 54]);
```

Check membership:

```dart
var ingredients = Set<String>();
ingredients.addAll(['gold', 'titanium', 'xenon']);
assert(ingredients.contains('titanium'));
assert(ingredients.containsAll(['titanium', 'xenon']));
```

Compute intersections:

```dart
var nobleGases = Set.from(['xenon', 'argon']);
var intersection = ingredients.intersection(nobleGases);
assert(intersection.contains('xenon'));
```

### Maps

Maps associate keys with values:

```dart
var hawaiianBeaches = {
  'Oahu': ['Waikiki', 'Kailua', 'Waimanalo'],
  'Big Island': ['Wailea Bay', 'Pololu Beach'],
};

var searchTerms = Map();
var nobleGases = Map<int, String>();
```

Access and modify entries:

```dart
var nobleGases = {54: 'xenon'};
assert(nobleGases[54] == 'xenon');
assert(nobleGases.containsKey(54));
nobleGases.remove(54);
```

Retrieve keys and values:

```dart
var keys = hawaiianBeaches.keys;
var values = hawaiianBeaches.values;
assert(keys.length == 3);
assert(values.length == 3);
```

Use `putIfAbsent()` for conditional assignment:

```dart
var teamAssignments = <String, String>{};
teamAssignments.putIfAbsent('Catcher', () => pickToughestKid());
assert(teamAssignments['Catcher'] != null);
```

### Common Collection Methods

Check if collections have items:

```dart
var coffees = <String>[];
var teas = ['green', 'black', 'chamomile', 'earl grey'];
assert(coffees.isEmpty);
assert(teas.isNotEmpty);
```

Iterate using `forEach()`:

```dart
teas.forEach((tea) => print('I drink $tea'));

hawaiianBeaches.forEach((k, v) {
  print('I want to visit $k and swim at $v');
});
```

Transform collections with `map()`:

```dart
var loudTeas = teas.map((tea) => tea.toUpperCase());
loudTeas.forEach(print);
var loudTeaList = teas.map((tea) => tea.toUpperCase()).toList();
```

Filter with conditions:

```dart
bool isDecaffeinated(String teaName) => teaName == 'chamomile';
var decaffeinatedTeas = teas.where((tea) => isDecaffeinated(tea));

assert(teas.any(isDecaffeinated));
assert(!teas.every(isDecaffeinated));
```

## URIs

The `Uri` class handles URI encoding and decoding.

### Encoding Full URIs

```dart
var uri = 'https://example.org/api?foo=some message';
var encoded = Uri.encodeFull(uri);
assert(encoded == 'https://example.org/api?foo=some%20message');
var decoded = Uri.decodeFull(encoded);
assert(uri == decoded);
```

### Encoding URI Components

```dart
var uri = 'https://example.org/api?foo=some message';
var encoded = Uri.encodeComponent(uri);
assert(encoded == 'https%3A%2F%2Fexample.org%2Fapi%3Ffoo%3Dsome%20message');
var decoded = Uri.decodeComponent(encoded);
assert(uri == decoded);
```

### Parsing URIs

```dart
var uri = Uri.parse('https://example.org:8080/foo/bar#frag');
assert(uri.scheme == 'https');
assert(uri.host == 'example.org');
assert(uri.path == '/foo/bar');
assert(uri.fragment == 'frag');
assert(uri.origin == 'https://example.org:8080');
```

### Building URIs

```dart
var uri = Uri(
  scheme: 'https',
  host: 'example.org',
  path: '/foo/bar',
  fragment: 'frag',
  queryParameters: {'lang': 'dart'},
);
assert(uri.toString() == 'https://example.org/foo/bar?lang=dart#frag');

var httpUri = Uri.http('example.org', '/foo/bar', {'lang': 'dart'});
var httpsUri = Uri.https('example.org', '/foo/bar', {'lang': 'dart'});
assert(httpUri.toString() == 'http://example.org/foo/bar?lang=dart');
assert(httpsUri.toString() == 'https://example.org/foo/bar?lang=dart');
```

## Dates and Times

Create `DateTime` objects:

```dart
var now = DateTime.now();
var y2k = DateTime(2000); // January 1, 2000
y2k = DateTime(2000, 1, 2); // January 2, 2000
y2k = DateTime.utc(2000); // UTC
y2k = DateTime.fromMillisecondsSinceEpoch(946684800000, isUtc: true);
y2k = DateTime.parse('2000-01-01T00:00:00Z');

var sameTimeLastYear = now.copyWith(year: now.year - 1);
```

Access milliseconds since epoch:

```dart
var y2k = DateTime.utc(2000);
assert(y2k.millisecondsSinceEpoch == 946684800000);
var unixEpoch = DateTime.utc(1970);
assert(unixEpoch.millisecondsSinceEpoch == 0);
```

Use `Duration` for date arithmetic:

```dart
var y2k = DateTime.utc(2000);
var y2001 = y2k.add(const Duration(days: 366));
assert(y2001.year == 2001);

var december2000 = y2001.subtract(const Duration(days: 30));
assert(december2000.year == 2000);
assert(december2000.month == 12);

var duration = y2001.difference(y2k);
assert(duration.inDays == 366);
```

## Utility Classes

### Comparing Objects

Implement `Comparable` to enable sorting:

```dart
class Line implements Comparable<Line> {
  final int length;
  const Line(this.length);

  @override
  int compareTo(Line other) => length - other.length;
}

void main() {
  var short = const Line(1);
  var long = const Line(100);
  assert(short.compareTo(long) < 0);
}
```

### Implementing Map Keys

Override `hashCode` and `==` for custom objects as map keys:

```dart
class Person {
  final String firstName, lastName;

  Person(this.firstName, this.lastName);

  @override
  int get hashCode => Object.hash(firstName, lastName);

  @override
  bool operator ==(Object other) {
    return other is Person &&
        other.firstName == firstName &&
        other.lastName == lastName;
  }
}

void main() {
  var p1 = Person('Bob', 'Smith');
  var p2 = Person('Bob', 'Smith');
  assert(p1.hashCode == p2.hashCode);
  assert(p1 == p2);
}
```

Use `Object.hashAll()` for collections and `Object.hashAllUnordered()` when order doesn't matter.

### Iteration

Extend `IterableBase` or implement `Iterable` to support iteration:

```dart
class Process {
  // Represents a process...
}

class ProcessIterator implements Iterator<Process> {
  @override
  Process get current => ...
  @override
  bool moveNext() => ...
}

class Processes extends IterableBase<Process> {
  @override
  final Iterator<Process> iterator = ProcessIterator();
}

void main() {
  for (final process in Processes()) {
    // Do something with the process.
  }
}
```

## Exceptions

Define custom exceptions by implementing `Exception`:

```dart
class FooException implements Exception {
  final String? msg;

  const FooException([this.msg]);

  @override
  String toString() => msg ?? 'FooException';
}
```

Common exceptions include `NoSuchMethodError` and `ArgumentError`.

## Weak References and Finalizers

- `WeakReference` stores references without affecting garbage collection
- `Expando` adds properties to objects without modification
- `Finalizer` executes callbacks after objects are no longer referenced
- `NativeFinalizer` provides stronger guarantees for native resource cleanup
- `Finalizable` interface prevents premature garbage collection

These features were added in Dart 2.17.
