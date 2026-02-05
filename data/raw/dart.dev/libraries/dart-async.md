# dart:async Library Documentation

## Overview

The `dart:async` library provides asynchronous programming tools in Dart. According to the documentation, "A Future is like a promise for a result to be provided sometime in the future. A Stream is a way to get a sequence of values, such as events."

### Import Statement

```dart
import 'dart:async';
```

Note: Future and Stream classes are available from `dart:core`, so explicit import isn't required for basic usage.

## Futures

Futures represent eventual results of asynchronous operations.

### Using await (Recommended)

Modern Dart favors `await` syntax over direct Future API calls:

```dart
Future<void> runUsingAsyncAwait() async {
  var entryPoint = await findEntryPoint();
  var exitCode = await runExecutable(entryPoint, args);
  await flushThenExit(exitCode);
}
```

Error handling with await:

```dart
try {
  var exitCode = await runExecutable(entryPoint, args);
} catch (e) {
  // Handle the error
}
```

### Basic Future Usage

The `then()` method executes code when a Future completes:

```dart
httpClient.read(url).then((String result) {
  print(result);
});
```

Handle errors with `catchError()`:

```dart
httpClient
    .read(url)
    .then((String result) {
      print(result);
    })
    .catchError((e) {
      // Handle or ignore the error
    });
```

### Chaining Multiple Async Methods

```dart
Future result = costlyQuery(url);
result
    .then((value) => expensiveWork(value))
    .then((_) => lengthyComputation())
    .then((_) => print('Done!'))
    .catchError((exception) {
      /* Handle exception... */
    });
```

Equivalent with await:

```dart
try {
  final value = await costlyQuery(url);
  await expensiveWork(value);
  await lengthyComputation();
  print('Done!');
} catch (e) {
  /* Handle exception... */
}
```

### Waiting for Multiple Futures

Use `Future.wait()` for parallel operations:

```dart
await Future.wait([
  deleteLotsOfFiles(),
  copyLotsOfFiles(),
  checksumLotsOfOtherFiles(),
]);
print('Done with all the long steps!');
```

### Handling Errors with Multiple Futures

When individual error handling matters, use wait on iterables or records:

**With iterables:**

```dart
try {
  var results = await [delete(), copy(), errorResult()].wait;
} on ParallelWaitError<List<bool?>, List<AsyncError?>> catch (e) {
  print(e.values[0]);  // Successful result
  print(e.errors[2]);  // Error from failed future
}
```

**With records (preserving types):**

```dart
try {
  final (deleteInt, copyString, errorBool) =
      await (delete(), copy(), errorResult()).wait;
} on ParallelWaitError<
  (int?, String?, bool?),
  (AsyncError?, AsyncError?, AsyncError?)
> catch (e) {
  // Handle errors
}
```

## Streams

Streams represent sequences of asynchronous events or data.

### Async For Loops (Recommended)

Modern approach to stream consumption:

```dart
void main(List<String> arguments) async {
  if (await FileSystemEntity.isDirectory(searchPath)) {
    final startingDir = Directory(searchPath);
    await for (final entity in startingDir.list()) {
      if (entity is File) {
        searchFile(entity, searchTerms);
      }
    }
  }
}
```

### Listening to Stream Data

Subscribe using `listen()`:

```dart
submitButton.onClick.listen((e) {
  submitData();
});
```

Access specific events:

```dart
// Single event methods
stream.first;
stream.last;
stream.single;

// Conditional selection
stream.firstWhere((event) => condition);
stream.skip(5);
stream.take(3);
stream.where((event) => condition);
```

### Transforming Stream Data

Transform stream types:

```dart
var lines = inputStream
    .transform(utf8.decoder)
    .transform(const LineSplitter());
```

### Error and Completion Handling

**With async for loops:**

```dart
Future<void> readFileAwaitFor() async {
  var config = File('config.txt');
  Stream<List<int>> inputStream = config.openRead();

  var lines = inputStream
      .transform(utf8.decoder)
      .transform(const LineSplitter());
  try {
    await for (final line in lines) {
      print('Got ${line.length} characters from stream');
    }
    print('file is now closed');
  } catch (e) {
    print(e);
  }
}
```

**With Stream API:**

```dart
inputStream
    .transform(utf8.decoder)
    .transform(const LineSplitter())
    .listen(
      (String line) {
        print('Got ${line.length} characters from stream');
      },
      onDone: () {
        print('file is now closed');
      },
      onError: (e) {
        print(e);
      },
    );
```

## Additional Resources

The documentation references several related tutorials covering async/await patterns, error handling, stream creation, and Dart concurrency concepts with isolates and event loops.
