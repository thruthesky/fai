# Libraries & Imports in Dart

## Overview

The `import` and `library` directives enable modular and shareable code organization. "Every Dart file (plus its parts) is a library, even if it doesn't use a library directive." Dart employs underscores for privacy rather than keywords like `public` or `private`, providing straightforward access control at the library level.

## Using Libraries

### Basic Import Syntax

Import a library using the `import` directive with a URI:

```dart
import 'dart:js_interop';
```

For package manager libraries:

```dart
import 'package:test/test.dart';
```

### Specifying Library Prefixes

When importing libraries with conflicting identifiers, apply a namespace prefix:

```dart
import 'package:lib1/lib1.dart';
import 'package:lib2/lib2.dart' as lib2;

Element element1 = Element();        // From lib1
lib2.Element element2 = lib2.Element(); // From lib2
```

The wildcard prefix `_` allows accessing non-private extensions without binding the import.

### Selective Imports

Use `show` to import specific members or `hide` to exclude them:

```dart
import 'package:lib1/lib1.dart' show foo;
import 'package:lib2/lib2.dart' hide foo;
```

### Deferred (Lazy) Loading

For web applications, defer library loading to reduce startup time:

```dart
import 'package:greetings/hello.dart' deferred as hello;

Future<void> greet() async {
  await hello.loadLibrary();
  hello.printGreeting();
}
```

**Key points about deferred loading:**
- Available for web targets only (Flutter has separate implementation)
- Constants in deferred libraries aren't constants in the importing file
- Cannot use deferred library types directly in importing file
- `loadLibrary()` returns a `Future` and loads only once

### Library Directive

Attach documentation or metadata to the library declaration:

```dart
/// A really great test library.
@TestOn('browser')
library;
```

## Implementing Libraries

For library creation guidance, consult the [Create Packages](/tools/pub/create-packages) documentation, which covers:

- Source code organization
- The `export` directive
- The `part` directive
- Conditional imports/exports for multi-platform support
