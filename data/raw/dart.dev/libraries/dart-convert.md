# dart:convert Library Documentation

## Overview

The `dart:convert` library provides converters for JSON and UTF-8 encoding/decoding, along with support for creating custom converters. JSON represents structured data in text format, while UTF-8 is a variable-width encoding supporting Unicode characters.

**Import statement:**
```dart
import 'dart:convert';
```

---

## Decoding and Encoding JSON

### Decoding JSON
Use `jsonDecode()` to convert JSON strings into Dart objects:

```dart
var jsonString = '''
  [
    {"score": 40},
    {"score": 80}
  ]
''';

var scores = jsonDecode(jsonString);
assert(scores is List);

var firstScore = scores[0];
assert(firstScore is Map);
assert(firstScore['score'] == 40);
```

### Encoding JSON
Use `jsonEncode()` to convert Dart objects to JSON-formatted strings:

```dart
var scores = [
  {'score': 40},
  {'score': 80},
  {'score': 100, 'overtime': true, 'special_guest': null},
];

var jsonText = jsonEncode(scores);
assert(
  jsonText ==
      '[{"score":40},{"score":80},'
          '{"score":100,"overtime":true,'
          '"special_guest":null}]',
);
```

**Supported types:** int, double, String, bool, null, List, and Map (string keys only). Collections are encoded recursively.

For non-directly-encodable objects, either pass a conversion function to `jsonEncode()` or implement a `toJson()` method.

---

## UTF-8 Character Encoding/Decoding

### Decoding UTF-8 Bytes
Convert UTF-8 encoded bytes to strings using `utf8.decode()`:

```dart
List<int> utf8Bytes = [
  0xc3, 0x8e, 0xc3, 0xb1, 0xc5, 0xa3, 0xc3, 0xa9,
  0x72, 0xc3, 0xb1, 0xc3, 0xa5, 0xc5, 0xa3, 0xc3,
  0xae, 0xc3, 0xb6, 0xc3, 0xb1, 0xc3, 0xa5, 0xc4,
  0xbc, 0xc3, 0xae, 0xc5, 0xbe, 0xc3, 0xa5, 0xc5,
  0xa3, 0xc3, 0xae, 0xe1, 0xbb, 0x9d, 0xc3, 0xb1,
];

var funnyWord = utf8.decode(utf8Bytes);
assert(funnyWord == 'Îñţérñåţîöñåļîžåţîờñ');
```

### Stream Decoding
Transform UTF-8 character streams using `utf8.decoder`:

```dart
var lines = utf8.decoder.bind(inputStream).transform(const LineSplitter());
try {
  await for (final line in lines) {
    print('Got ${line.length} characters from stream');
  }
  print('file is now closed');
} catch (e) {
  print(e);
}
```

### Encoding to UTF-8
Convert Dart strings to UTF-8 bytes using `utf8.encode()`:

```dart
Uint8List encoded = utf8.encode('Îñţérñåţîöñåļîžåţîờñ');

assert(encoded.length == utf8Bytes.length);
for (int i = 0; i < encoded.length; i++) {
  assert(encoded[i] == utf8Bytes[i]);
}
```

---

## Additional Features

The library also includes converters for ASCII and ISO-8859-1 (Latin1) encoding. Refer to the [official API reference](https://api.dart.dev/dart-convert/dart-convert-library.html) for complete details.
