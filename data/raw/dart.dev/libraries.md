# Dart's Core Libraries

## Overview

Dart provides a comprehensive collection of core libraries supporting essential programming tasks. These include working with collections (`dart:collection`), mathematical operations (`dart:math`), and data encoding/decoding (`dart:convert`). Additional functionality is available through commonly used packages.

## Library Tour

The main guides cover major features of Dart's core libraries, offering foundational knowledge rather than exhaustive documentation. The Dart API reference contains detailed information about library members.

### Primary Libraries

**dart:core**
"Built-in types, collections, and other core functionality. This library is automatically imported into every Dart program."

**dart:async**
Provides asynchronous programming support through `Future` and `Stream` classes.

**dart:math**
Mathematical constants, functions, and random number generation capabilities.

**dart:convert**
"Encoders and decoders for converting between different data representations, including JSON and UTF-8."

**dart:io**
"I/O for programs that can use the Dart VM, including Flutter apps, servers, and command-line scripts."

**dart:js_interop**
"APIs for interop with the web platform. Along with `package:web`, `dart:js_interop` replaces `dart:html`."

## Multi-Platform Libraries

Libraries available across all Dart platforms include:

- `dart:core` – fundamental functionality
- `dart:async` and `package:async` – asynchronous operations
- `dart:collection` and `package:collection` – collection utilities
- `dart:convert` and `package:convert` – data conversion
- `dart:developer` – debugger and inspector interaction
- `dart:math` – mathematical operations
- `dart:typed_data` and `package:typed_data` – efficient fixed-size data handling

## Native Platform Libraries

For AOT- and JIT-compiled code:

- `dart:ffi` and `package:ffi` – foreign function interfaces
- `dart:io` and `package:io` – file and network operations
- `dart:isolate` – concurrent programming
- `dart:mirrors` – reflection capabilities (Native JIT only)

## Web Platform Libraries

**Current (recommended):**

- `package:web` – browser API bindings
- `dart:js_interop` – JavaScript interoperability
- `dart:js_interop_unsafe` – dynamic JavaScript object manipulation

**Legacy (deprecated):**

- `dart:html`, `dart:indexed_db`, `dart:svg`, `dart:web_audio`, `dart:web_gl`
- `dart:js`, `dart:js_util`, and `package:js`

The documentation recommends transitioning to `package:web` for modern web development.
