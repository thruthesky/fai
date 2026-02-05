# Effective Dart: Usage

## Overview

This guide provides best practices for using Dart language features in maintainable code.

## Libraries

### DO use strings in part of directives

```dart
part of '../../my_library.dart';
```

### DON'T import libraries inside another package's src directory

The src directory contains private implementation code.

### PREFER relative import paths

Within lib, use relative imports when they don't cross directory boundaries.

## Null Safety

### DON'T explicitly initialize variables to null

Nullable variables are implicitly initialized to null.

### DON'T compare non-nullable booleans to true or false

```dart
if (nonNullableBool) { ... }
if (!nonNullableBool) { ... }
```

## Strings

### DO use adjacent strings for concatenating literals

```dart
raiseAlarm(
  'ERROR: Parts are on fire. '
  'Other parts have martians.',
);
```

### PREFER interpolation over concatenation

```dart
'Hello, $name! You are ${year - birth} years old.';
```

## Collections

### DO use collection literals

```dart
var points = <Point>[];
var addresses = <String, Address>{};
var counts = <int>{};
```

### DON'T use .length to check if empty

```dart
if (lunchBox.isEmpty) return 'hungry...';
if (words.isNotEmpty) return words.join(' ');
```

## Functions

### DO use function declarations to bind functions to names

### DON'T create lambdas when tear-offs work

```dart
charCodes.forEach(print);
var strings = charCodes.map(String.fromCharCode);
```

## Constructors

### DO use initializing formals

```dart
class Point {
  double x, y;
  Point(this.x, this.y);
}
```

### DON'T use new

The keyword is optional and deprecated.

### DON'T use const redundantly

The keyword is implicit in const contexts.

## Error Handling

### AVOID catches without on clauses

```dart
try {
  somethingRisky();
} on DownloadException catch (e) {
  handle(e);
}
```

### DO use rethrow to preserve stack traces

```dart
try {
  somethingRisky();
} catch (e) {
  if (!canHandle(e)) rethrow;
  handle(e);
}
```

## Asynchronous Programming

### PREFER async/await over raw futures

```dart
Future<int> countActivePlayers(String teamName) async {
  try {
    var team = await downloadTeam(teamName);
    if (team == null) return 0;
    var players = await team.roster;
    return players.where((p) => p.isActive).length;
  } on DownloadException catch (e) {
    log.error(e);
    return 0;
  }
}
```

---

## Source

- **URL**: https://dart.dev/effective-dart/usage
- **Fetched**: 2026-01-27
