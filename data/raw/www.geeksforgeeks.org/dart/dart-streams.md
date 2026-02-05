# Dart Streams: Comprehensive Guide

## What is a Stream?

A stream represents a sequence of asynchronous events where the stream notifies listeners when data becomes available, rather than listeners requesting it on-demand. Streams deliver data sequentially and can emit data events, error events, or completion signals.

## Key Advantages

The primary benefit of streams is maintaining loose coupling in code. Data producers emit values independently without knowing consumers, while consumers need only follow the stream interface. The underlying data generation mechanism remains completely abstracted.

## Core Concepts

**Two Processing Methods:**
- `await for` loops
- `listen()` from the Stream API

**Stream Analogy:** Think of streams like pipes—values enter one end, and listeners receive them at the other end.

## Creating Streams

### Basic Example with await for

```dart
Future<int> sumStream(Stream<int> stream) async {
  var sum = 0;
  await for(var value in stream) {
    sum += value;
  }
  return sum;
}

Future<void> main() async {
  final stream = Stream<int>.fromIterable([1,2,3,4,5]);
  final sum = await sumStream(stream);
  print('Sum: $sum');
}
```

The `async` keyword is required when using `await for` loops.

## Important Flutter Concepts

### StreamController

A `StreamController` simplifies stream management by automatically creating both a stream and sink. It enables controlling stream behavior and checking properties like subscriber count or pause status.

**Key Methods:**
- `add()` - forwards data to the sink
- `addError()` - notifies listeners of errors
- `listen()` - establishes stream listeners

### StreamBuilder Widget

A Flutter widget that rebuilds UI when stream values update. It requires two parameters:
- **stream** - method returning a stream object
- **builder** - widgets for different StreamBuilder states

## Practical Implementation: Countdown App

```dart
import 'dart:async';
import 'package:flutter/material.dart';

class _CounterAppState extends State<CounterApp> {
  StreamController _controller = StreamController();
  int _counter = 10;

  void StartTimer() async {
    Timer.periodic(Duration(seconds: 1), (timer) {
      _counter--;
      _controller.sink.add(_counter);

      if(_counter <= 0){
        timer.cancel();
        _controller.close();
      }
    });
  }

  @override
  void dispose() {
    super.dispose();
    _controller.close();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            StreamBuilder(
              initialData: _counter,
              stream: _controller.stream,
              builder: (context, snapshot) {
                return Text('${snapshot.data}');
              }
            ),
            ElevatedButton(
              onPressed: () => StartTimer(),
              child: Text('Start Count Down')
            )
          ],
        ),
      ),
    );
  }
}
```

## Stream Components

**Sink:** Entry point for adding data into the stream pipe

**Source:** Point from which listeners receive stream data

## Two Stream Types

### 1. Single Subscription Streams (Default)

- Can only be listened to once
- Don't generate events until having a listener
- Stop sending when listener stops, even if data exists
- Ideal for single-use operations like file downloads

```dart
StreamController<String> controller = StreamController<String>();
Stream<String> stream = controller.stream;

void main() {
  StreamSubscription<String> subscriber = stream.listen(
    (String data) => print(data),
    onError: (error) => print(error),
    onDone: () => print('Stream closed!')
  );

  controller.sink.add('GeeksforGeeks!');
  controller.addError('Error!');
  controller.close();
}
```

### 2. Broadcast Streams

- Allow multiple listeners simultaneously
- Fire events regardless of listener presence
- Created via `asBroadcastStream()` on existing streams

**Syntax:** `final broadcastStream = singleStream.asBroadcastStream();`

```dart
StreamController<String> controller =
  StreamController<String>.broadcast();

void main() {
  StreamSubscription<String> subscriber1 = controller.stream.listen(
    (String data) => print('Subscriber1: $data')
  );

  StreamSubscription<String> subscriber2 = controller.stream.listen(
    (String data) => print('Subscriber2: $data')
  );

  controller.sink.add('GeeksforGeeks!');
  controller.close();
}
```

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-streams/
- **Fetched**: 2026-01-27
