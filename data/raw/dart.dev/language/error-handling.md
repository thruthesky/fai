# Error Handling in Dart

## Exceptions

Dart code can throw and catch exceptions to handle unexpected situations. Unlike Java, all Dart exceptions are unchecked—"Methods don't declare which exceptions they might throw, and you aren't required to catch any exceptions."

The language provides `Exception` and `Error` types with predefined subtypes, though you can define custom exceptions or throw any non-null object.

### Throw

To raise an exception:

```dart
throw FormatException('Expected at least 1 section');
```

You can also throw arbitrary objects:

```dart
throw 'Out of llamas!';
```

Throwing works in arrow functions and expressions:

```dart
void distanceTo(Point other) => throw UnimplementedError();
```

### Catch

Catching an exception prevents it from propagating and allows handling:

```dart
try {
  breedMoreLlamas();
} on OutOfLlamasException {
  buyMoreLlamas();
}
```

For multiple exception types, specify multiple catch clauses. The first matching clause handles the exception:

```dart
try {
  breedMoreLlamas();
} on OutOfLlamasException {
  buyMoreLlamas();
} on Exception catch (e) {
  print('Unknown exception: $e');
} catch (e) {
  print('Something really unknown: $e');
}
```

Use `on` to specify exception type; use `catch` to access the exception object. You can optionally capture the stack trace:

```dart
try {
  // code
} catch (e, s) {
  print('Exception: $e');
  print('Stack trace: $s');
}
```

Use `rethrow` to partially handle an exception while allowing propagation.

### Finally

The `finally` clause runs whether an exception occurs or not:

```dart
try {
  breedMoreLlamas();
} finally {
  cleanLlamaStalls();
}
```

Finally executes after matching catch clauses complete.

## Assert

During development, assertions disrupt execution if conditions fail:

```dart
assert(text != null);
assert(number < 100);
assert(urlString.startsWith('https'));
```

Add a message to assertions:

```dart
assert(
  urlString.startsWith('https'),
  'URL ($urlString) should start with "https".',
);
```

Assertions succeed if the condition is true; they throw `AssertionError` if false. "In production code, assertions are ignored, and the arguments to `assert` aren't evaluated."
