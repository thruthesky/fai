# Asynchronous Programming in Dart

## Overview

Dart provides comprehensive support for asynchronous programming through `Future` and `Stream` objects. The `async` and `await` keywords enable developers to write asynchronous code that resembles synchronous code in structure.

## Handling Futures

### Understanding Futures

Functions returning `Future` or `Stream` objects are asynchronous - they initiate potentially time-consuming operations like I/O without blocking execution. Two approaches exist for working with completed Futures:

1. Using `async` and `await` keywords
2. Using the Future API from `dart:async`

### Using await and async

Code implementing `await` must reside within an `async` function:

```dart
Future<void> checkVersion() async {
  var version = await lookUpVersion();
  // Do something with version
}
```

**Key behavior**: An `async` function executes only until encountering its first `await` expression, then returns a `Future` object, resuming afterward.

### Error Handling with Futures

Use traditional exception handling:

```dart
try {
  version = await lookUpVersion();
} catch (e) {
  // React to inability to look up the version
}
```

### Multiple await Expressions

Sequential waiting is supported:

```dart
var entrypoint = await findEntryPoint();
var exitCode = await runExecutable(entrypoint, args);
await flushThenExit(exitCode);
```

**Important note**: If the expression is not a Future, Dart automatically wraps it as one. The `await` expression pauses until the result becomes available.

### Using await in main()

The application entry point requires marking:

```dart
void main() async {
  checkVersion();
  print('In main: version is ${await lookUpVersion()}');
}
```

## Declaring async Functions

### Function Declaration Pattern

Adding `async` modifies the return type to `Future`. Compare:

**Synchronous:**

```dart
String lookUpVersion() => '1.0.0';
```

**Asynchronous:**

```dart
Future<String> lookUpVersion() async => '1.0.0';
```

Functions returning no useful value should use `Future<void>` as the return type.

## Handling Streams

### Asynchronous for Loops

Process stream values using `await for`:

```dart
await for (varOrType identifier in expression) {
  // Executes each time the stream emits a value.
}
```

The expression must have type `Stream`. Execution follows this sequence:

1. Wait for stream emission
2. Execute loop body with the emitted value
3. Repeat until stream closes

### Stream Loop Control

Break or return statements terminate iteration and unsubscribe from the stream.

### Stream Processing in main()

```dart
void main() async {
  // ...
  await for (final request in requestServer) {
    handleRequest(request);
  }
  // ...
}
```

**Usage caution**: Avoid `await for` with infinite streams like UI event listeners, as it blocks until completion.

## Additional Resources

- Interactive tutorial: Asynchronous Programming Tutorial
- Detailed reference: `dart:async` Library
- Linter rule for best practices: `unawaited_futures`
