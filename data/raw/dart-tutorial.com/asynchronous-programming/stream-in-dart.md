# Streams in Dart

## Overview

A stream represents "a sequence of asynchronous events representing multiple values that will arrive in the future." Streams handle sequences of events rather than single occurrences. One stream can have multiple listeners, and all recipients obtain identical values.

The pipe analogy illustrates this concept: values enter one end, and listeners on the opposite end receive those values. These values may be of any type, errors, or a completion event.

| Type | Single Value | Zero or More Values |
|------|--------------|-------------------|
| Sync | `int` | `Iterator` |
| Async | `Future<int>` | `Stream<int>` |

## Creating Streams

### Using `async*` and `yield`

```dart
Stream<String> getUserName() async* {
  await Future.delayed(Duration(seconds: 1));
  yield 'Mark';
  await Future.delayed(Duration(seconds: 1));
  yield 'John';
  await Future.delayed(Duration(seconds: 1));
  yield 'Smith';
}
```

**Note:** The `yield` keyword emits stream values and requires `async*` syntax.

### Using `Stream.fromIterable()`

```dart
Stream<String> getUserName() {
  return Stream.fromIterable(['Mark', 'John', 'Smith']);
}
```

This returns results immediately.

## Using Streams

Streams are consumed with the `await for` loop:

```dart
void main() async {
  await for (String name in getUserName()) {
    print(name);
  }
}
```

**Output:**
```
Mark
John
Smith
```

## Future vs Stream

| Aspect | Future | Stream |
|--------|--------|--------|
| Return Type | Single result over time | Zero or more values |
| Listening | Cannot monitor variable changes | Can listen to variable modifications |
| Builder | FutureBuilder | StreamBuilder |
| Syntax | `Future<data_type>` | `Stream<data_type>` |

## Stream Types

### Single Subscription Streams

By default, streams support single subscriptions. They retain values until subscription and can only be observed once. Attempting multiple subscriptions raises an exception. All event values must arrive in correct sequence without omission.

### Broadcast Streams

These facilitate multiple subscriptions and can be listened to many times. Use broadcast streams for scenarios requiring multiple observers. Syntax:

```dart
StreamController<data_type> controller = StreamController<data_type>.broadcast();
```

## Stream Management

### Creating StreamController

```dart
StreamController<data_type> controller = StreamController<data_type>();
Stream stream = controller.stream;
```

### Subscribing

```dart
stream.listen((value) {
  print("Value from controller: $value");
});
```

### Adding Values

```dart
controller.add(3);
```

**Output:**
```
Value from controller: 3
```

### Managing Subscriptions

```dart
StreamSubscription<int> streamSubscription = stream.listen((value){
  print("Value from controller: $value");
});
```

### Canceling Streams

```dart
streamSubscription.cancel();
```

## Core Classes

**Stream:** Represents asynchronous data sequences.

**EventSink:** Operates like streams flowing in reverse direction.

**StreamController:** Streamlines stream administration, automatically producing stream and sink plus methods for controlling behavior.

**StreamSubscription:** Maintains subscription references, enabling pause, resume, or cancellation operations.

## Stream Methods

### `listen()`

Returns a `StreamSubscription` object enabling pause, resume after pause, or complete cancellation.

```dart
final subscription = myStream.listen()
```

### `onError()`

Catches and processes errors:

```dart
onError: (err){
  // handle error
}
```

### `cancelOnError`

Defaults to true but can be set false to preserve subscription despite errors:

```dart
cancelOnError : false
```

### `onDone()`

Executes code when the stream finishes transmitting data:

```dart
onDone: (){
  // completion logic
}
```

## Keywords

**`async*`:** Works similarly to `async` for futures within streams.

**`yield`:** Emits values from generators (async or sync). Returns values from Iterable or Stream collections.

**`yield*`:** Recursively calls Iterable or Stream functions.

## Code Examples

### Example 1: Basic StreamController

```dart
import 'dart:async';

void main() {
  var controller = StreamController();
  controller.stream.listen((event) {
    print(event);
  });
  controller.add('Hello');
  controller.add(42);
  controller.addError('Error!');
  controller.close();
}
```

**Output:**
```
Hello
42
Uncaught Error: Error!
```

### Example 2: Stream with Numbers

```dart
Stream<int> numberOfStream(int number) async* {
  for (int i = 0; i <= number; i++) {
    yield i;
  }
}

void main(List<String> arguments) {
  var stream = numberOfStream(6);
  stream.listen((s) => print(s));
}
```

**Output:**
```
0
1
2
3
4
5
6
```

### Example 3: Delayed Stream Output

```dart
Stream<int> str(int n) async* {
 for (var i = 1; i <= n; i++) {
   await Future.delayed(Duration(seconds: 1));
   yield i;
 }
}

void main() {
 str(10).forEach(print);
}
```

**Output:**
```
1
2
3
4
5
```

### Example 4: Recursive Yield

```dart
Stream<int> str(int n) async* {
 if (n > 0) {
   await Future.delayed(Duration(seconds: 2));
   yield n;
   yield* str(n - 2);
 }
}

void main() {
 str(10).forEach(print);
}
```

**Output:**
```
10
8
6
4
2
```

## Keyword Comparisons

### `async` vs `async*`

| Feature | async | async* |
|---------|-------|--------|
| Return Type | Future | Stream |
| Purpose | Long-running operations | Multiple future values |
| Result Wrapping | Future-wrapped | Stream-wrapped |

### `yield` vs `yield*`

| Feature | yield | yield* |
|---------|-------|--------|
| Purpose | Returns single value to sequence | Returns from recursive generator |
| Continuation | Generator function continues | Enables recursive calls |

## Summary

Streams enable flexible, effective handling of asynchronous data flows. They process data as availability occurs rather than awaiting full loading. Common applications include monitoring real-time user interactions and receiving server data progressively. Implementation relies on Stream and StreamController classes for creation and administration.

---

## Source

- **URL**: https://dart-tutorial.com/asynchronous-programming/stream-in-dart/
- **Fetched**: 2026-01-27
