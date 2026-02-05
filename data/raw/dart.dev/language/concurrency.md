# Concurrency in Dart

## Overview

Dart enables concurrent programming through two main mechanisms: asynchronous APIs and isolates. Concurrent programming in Dart refers to both asynchronous APIs, like `Future` and `Stream`, and isolates, which allow you to move processes to separate cores.

## Event Loop

The event loop serves as Dart's runtime foundation. It processes events sequentially from a queue, handling everything from UI repaints to user input and I/O operations.

The conceptual event loop functions similarly to:

```dart
while (eventQueue.waitForEvent()) {
  eventQueue.processNextEvent();
}
```

This single-threaded model supports multi-tasking through asynchronous APIs that register callbacks for future execution.

## Asynchronous Programming

### Futures

A `Future` represents a placeholder for a value that will eventually complete:

```dart
Future<String> _readFileAsync(String filename) {
  final file = File(filename);
  return file.readAsString().then((contents) {
    return contents.trim();
  });
}
```

### Async-Await Syntax

The `async` and `await` keywords provide cleaner asynchronous syntax:

```dart
void main() async {
  final fileData = await _readFileAsync();
  final jsonData = jsonDecode(fileData);
  print('Number of JSON keys: ${jsonData.length}');
}

Future<String> _readFileAsync() async {
  final file = File(filename);
  final contents = await file.readAsString();
  return contents.trim();
}
```

The `await` keyword works only in functions that have `async` before the function body.

### Streams

Streams provide repeated values over time. A periodic stream example:

```dart
Stream<int> stream = Stream.periodic(const Duration(seconds: 1), (i) => i * i);
```

Use `await-for` loops and `yield` to process streams:

```dart
Stream<int> sumStream(Stream<int> stream) async* {
  var sum = 0;
  await for (final value in stream) {
    yield sum += value;
  }
}
```

## Isolates

Instead of threads, all Dart code runs inside isolates. Using isolates, your Dart code can perform multiple independent tasks at once, using additional processor cores if they're available.

### Main Isolate

Programs execute within a main isolate by default - the starting thread for execution and event handling.

### Isolate Lifecycle

Every isolate executes initialization code (like `main()`), optionally handles events, then exits.

### Background Workers

Offload computations to worker isolates to prevent UI freezing:

```dart
int slowFib(int n) => n <= 1 ? 1 : slowFib(n - 1) + slowFib(n - 2);

void fib40() async {
  var result = await Isolate.run(() => slowFib(40));
  print('Fib(40) = $result');
}
```

### Using Isolates

Two primary approaches exist:

**`Isolate.run()`** - Execute single computations on separate threads (recommended for most cases)

**`Isolate.spawn()`** - Create long-lived isolates handling multiple messages

### Limitations

#### Isolates Aren't Threads

Each isolate has its own state, ensuring that none of the state in an isolate is accessible from any other isolate. Global variables remain separate across isolates.

#### Message Types

Most Dart objects transmit via messages, but exceptions include `Socket`, `ReceivePort`, `DynamicLibrary`, `Pointer`, and objects marked `@pragma('vm:isolate-unsendable')`.

#### Synchronous Communication

Isolates can only communicate synchronously outside of pure Dart, using C code via FFI to do so.

### Performance Considerations

`Isolate.spawn()` allows performance optimizations through isolate groups - shared executable code between isolates. `Isolate.spawnUri()` operates slower and creates separate isolate groups.

## Concurrency on the Web

The Dart web platform lacks isolate support but provides web workers for background threading. Web workers' functionality and capabilities differ somewhat from isolates. Web workers copy data during transfers, while isolates can more efficiently transfer memory.

## Additional Resources

- Actor model documentation
- API references: `Isolate.exit()`, `Isolate.spawn()`, `ReceivePort`, `SendPort`
- Flutter's `IsolateNameServer` for managing multiple isolates
- Package: `isolate_name_server` for non-Flutter applications
