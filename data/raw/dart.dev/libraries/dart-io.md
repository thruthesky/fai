# dart:io Library Documentation

## Overview

The `dart:io` library provides APIs for handling files, directories, processes, sockets, WebSockets, and HTTP clients and servers. As stated in the documentation: "Only non-web Flutter apps, command-line scripts, and servers can import and use dart:io, not web apps."

The library emphasizes asynchronous operations to prevent blocking applications. Most operations return results through `Future` or `Stream` objects rather than synchronous calls.

## Files and Directories

### Reading Text Files

You can read entire text files or line-by-line using these approaches:

```dart
void main() async {
  var config = File('config.txt');

  // Read entire file
  var stringContents = await config.readAsString();
  print('The file is ${stringContents.length} characters long.');

  // Read line by line
  var lines = await config.readAsLines();
  print('The file is ${lines.length} lines long.');
}
```

### Reading Binary Files

For binary data, use `readAsBytes()`:

```dart
void main() async {
  var config = File('config.txt');
  var contents = await config.readAsBytes();
  print('The file is ${contents.length} bytes long.');
}
```

### Error Handling

Wrap file operations in try-catch blocks:

```dart
void main() async {
  var config = File('config.txt');
  try {
    var contents = await config.readAsString();
    print(contents);
  } catch (e) {
    print(e);
  }
}
```

### Streaming File Contents

For large files, use streams with `await for`:

```dart
import 'dart:io';
import 'dart:convert';

void main() async {
  var config = File('config.txt');
  Stream<List<int>> inputStream = config.openRead();

  var lines = utf8.decoder.bind(inputStream)
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

### Writing File Contents

Use `IOSink` to write data:

```dart
var logFile = File('log.txt');
var sink = logFile.openWrite();
sink.write('FILE ACCESSED ${DateTime.now()}\n');
await sink.flush();
await sink.close();
```

For appending data:

```dart
var sink = logFile.openWrite(mode: FileMode.append);
```

### Listing Directory Contents

```dart
void main() async {
  var dir = Directory('tmp');

  try {
    var dirList = dir.list();
    await for (final FileSystemEntity f in dirList) {
      if (f is File) {
        print('Found file ${f.path}');
      } else if (f is Directory) {
        print('Found dir ${f.path}');
      }
    }
  } catch (e) {
    print(e.toString());
  }
}
```

### Other File/Directory Operations

- Creating: `create()` method
- Deleting: `delete()` method
- File length: `length()` method
- Random file access: `open()` method

## HTTP Clients and Servers

### HTTP Server

The `HttpServer` class enables building web servers. Example:

```dart
void main() async {
  final requests = await HttpServer.bind('localhost', 8888);
  await for (final request in requests) {
    processRequest(request);
  }
}

void processRequest(HttpRequest request) {
  print('Got request for ${request.uri.path}');
  final response = request.response;
  if (request.uri.path == '/dart') {
    response
      ..headers.contentType = ContentType('text', 'plain')
      ..write('Hello from the server');
  } else {
    response.statusCode = HttpStatus.notFound;
  }
  response.close();
}
```

### HTTP Client

Rather than using `HttpClient` directly, the documentation recommends using "a higher-level library like package:http" for making HTTP requests, as the native implementation is platform-dependent.

## Additional Resources

The library also provides APIs for processes, sockets, and WebSockets. Complete documentation is available in the [official dart:io API reference](https://api.dart.dev/dart-io/dart-io-library.html).
