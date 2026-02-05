# Isolates in Dart

## Overview

Isolates enable concurrent execution in Dart by allowing code to run in separate threads with independent memory. According to the documentation, "You should use isolates whenever your application is handling computations that are large enough to temporarily block other computations."

### Common Use Cases

Isolates prove beneficial for:
- Processing large JSON datasets
- Image, audio, and video manipulation
- Complex searching and filtering operations
- Database communications
- High-volume network requests

## Simple Worker Isolates

### Using `Isolate.run()`

The `Isolate.run()` method simplifies isolate management by handling spawning, execution, result capture, and cleanup automatically.

**Example: Running an existing method**

```dart
const String filename = 'with_keys.json';

void main() async {
  final jsonData = await Isolate.run(_readAndParseJson);
  print('Number of JSON keys: ${jsonData.length}');
}

Future<Map<String, dynamic>> _readAndParseJson() async {
  final fileData = await File(filename).readAsString();
  final jsonData = jsonDecode(fileData) as Map<String, dynamic>;
  return jsonData;
}
```

**Example: Using closures**

```dart
void main() async {
  final jsonData = await Isolate.run(() async {
    final fileData = await File(filename).readAsString();
    final jsonData = jsonDecode(fileData) as Map<String, dynamic>;
    return jsonData;
  });
  print('Number of JSON keys: ${jsonData.length}');
}
```

## Long-Lived Isolates with Ports

For repeated computations, long-lived isolates offer better performance than repeatedly spawning short-lived ones.

### ReceivePort and SendPort

These classes enable bidirectional communication:
- **ReceivePort**: Receives messages from other isolates
- **SendPort**: Sends messages to a specific ReceivePort

Each ReceivePort has an associated SendPort, but multiple SendPorts can target one ReceivePort.

### Basic Ports Example

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:isolate';

void main() async {
  final worker = Worker();
  await worker.spawn();
  await worker.parseJson('{"key":"value"}');
}

class Worker {
  late SendPort _sendPort;
  final Completer<void> _isolateReady = Completer.sync();

  Future<void> spawn() async {
    final receivePort = ReceivePort();
    receivePort.listen(_handleResponsesFromIsolate);
    await Isolate.spawn(_startRemoteIsolate, receivePort.sendPort);
  }

  void _handleResponsesFromIsolate(dynamic message) {
    if (message is SendPort) {
      _sendPort = message;
      _isolateReady.complete();
    } else if (message is Map<String, dynamic>) {
      print(message);
    }
  }

  static void _startRemoteIsolate(SendPort port) {
    final receivePort = ReceivePort();
    port.send(receivePort.sendPort);

    receivePort.listen((dynamic message) async {
      if (message is String) {
        final transformed = jsonDecode(message);
        port.send(transformed);
      }
    });
  }

  Future<void> parseJson(String message) async {
    await _isolateReady.future;
    _sendPort.send(message);
  }
}
```

### Robust Ports Example

This enhanced version adds error handling, message sequencing, and port closure:

```dart
import 'dart:async';
import 'dart:convert';
import 'dart:isolate';

void main() async {
  final worker = await Worker.spawn();
  print(await worker.parseJson('{"key":"value"}'));
  print(await worker.parseJson('"banana"'));
  print(await worker.parseJson('[true, false, null, 1, "string"]'));
  print(
    await Future.wait(
      [worker.parseJson('"yes"'), worker.parseJson('"no"')]
    ),
  );
  worker.close();
}

class Worker {
  final SendPort _commands;
  final ReceivePort _responses;
  final Map<int, Completer<Object?>> _activeRequests = {};
  int _idCounter = 0;
  bool _closed = false;

  Future<Object?> parseJson(String message) async {
    if (_closed) throw StateError('Closed');
    final completer = Completer<Object?>.sync();
    final id = _idCounter++;
    _activeRequests[id] = completer;
    _commands.send((id, message));
    return await completer.future;
  }

  static Future<Worker> spawn() async {
    final initPort = RawReceivePort();
    final connection = Completer<(ReceivePort, SendPort)>.sync();
    initPort.handler = (initialMessage) {
      final commandPort = initialMessage as SendPort;
      connection.complete((
        ReceivePort.fromRawReceivePort(initPort),
        commandPort,
      ));
    };

    try {
      await Isolate.spawn(_startRemoteIsolate, (initPort.sendPort));
    } on Object {
      initPort.close();
      rethrow;
    }

    final (ReceivePort receivePort, SendPort sendPort) =
        await connection.future;
    return Worker._(receivePort, sendPort);
  }

  Worker._(this._responses, this._commands) {
    _responses.listen(_handleResponsesFromIsolate);
  }

  void _handleResponsesFromIsolate(dynamic message) {
    final (int id, Object? response) = message as (int, Object?);
    final completer = _activeRequests.remove(id)!;

    if (response is RemoteError) {
      completer.completeError(response);
    } else {
      completer.complete(response);
    }

    if (_closed && _activeRequests.isEmpty) _responses.close();
  }

  static void _handleCommandsToIsolate(
    ReceivePort receivePort,
    SendPort sendPort,
  ) {
    receivePort.listen((message) {
      if (message == 'shutdown') {
        receivePort.close();
        return;
      }
      final (int id, String jsonText) = message as (int, String);
      try {
        final jsonData = jsonDecode(jsonText);
        sendPort.send((id, jsonData));
      } catch (e) {
        sendPort.send((id, RemoteError(e.toString(), '')));
      }
    });
  }

  static void _startRemoteIsolate(SendPort sendPort) {
    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);
    _handleCommandsToIsolate(receivePort, sendPort);
  }

  void close() {
    if (!_closed) {
      _closed = true;
      _commands.send('shutdown');
      if (_activeRequests.isEmpty) _responses.close();
      print('--- port closed ---');
    }
  }
}
```

## Key Concepts

**Memory Transfer**: When using `Isolate.run()`, results are transferred rather than copied between isolates, improving performance.

**Message Ordering**: The robust example uses message IDs and completers to ensure responses correlate with original requests, preventing ordering issues during concurrent operations.

**Error Handling**: The `RemoteError` class communicates exceptions from worker isolates back to the main isolate.

## Platform Notes

Flutter web does not support multiple isolates. Flutter applications should use the `compute` function instead of `Isolate.run()`.
