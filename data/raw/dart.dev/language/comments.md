# Comments in Dart

Dart supports three distinct comment types for code documentation and notes.

## Single-Line Comments

Single-line comments begin with `//` and extend to the end of the line. The Dart compiler ignores everything after `//`.

```dart
void main() {
  // TODO: refactor into an AbstractLlamaGreetingFactory?
  print('Welcome to my Llama farm!');
}
```

## Multi-Line Comments

Multi-line comments start with `/*` and end with `*/`. The compiler disregards all text between these delimiters (except documentation comments). These comments support nesting.

```dart
void main() {
  /*
   * This is a lot of work. Consider raising chickens.

  Llama larry = Llama();
  larry.feed();
  larry.exercise();
  larry.clean();
   */
}
```

## Documentation Comments

Documentation comments use `///` for single lines or `/**` for multi-line formats. Consecutive `///` lines function identically to block documentation comments.

Within documentation comments, bracketed text references program elements. The analyzer resolves names in brackets to classes, methods, fields, variables, functions, and parameters within the documented element's lexical scope.

```dart
/// A domesticated South American camelid (Lama glama).
///
/// Andean cultures have used llamas as meat and pack
/// animals since pre-Hispanic times.
///
/// Just like any other animal, llamas need to eat,
/// so don't forget to [feed] them some [Food].
class Llama {
  String? name;

  /// Feeds your llama [food].
  ///
  /// The typical llama eats one bale of hay per week.
  void feed(Food food) {
    // ...
  }

  /// Exercises your llama with an [activity] for
  /// [timeLimit] minutes.
  void exercise(Activity activity, int timeLimit) {
    // ...
  }
}
```

The `dart doc` tool parses Dart code and generates HTML documentation, converting bracketed references like `[feed]` into hyperlinks. See the [Dart API documentation](https://api.dart.dev) for examples and [Effective Dart: Documentation](/effective-dart/documentation) for best practices on structuring comments.
