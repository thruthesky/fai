# Metadata in Dart

## Overview

Metadata provides supplementary static information about code through annotations. These annotations begin with `@` and reference either compile-time constants or constant constructor calls. They can be attached to most Dart program constructs before their declarations.

## Built-in Annotations

### @Deprecated
Marks a declaration as deprecated with a message explaining the replacement and potential removal date.

Specific usage annotations include:
- `@Deprecated.extend()` – extending is deprecated
- `@Deprecated.implement()` – implementing is deprecated
- `@Deprecated.subclass()` – subclassing is deprecated
- `@Deprecated.mixin()` – mixing in is deprecated
- `@Deprecated.instantiate()` – instantiation is deprecated
- `@Deprecated.optional()` – omitting arguments is deprecated

**Example:**
```dart
class Television {
  /// Use [turnOn] to turn the power on instead.
  @Deprecated('Use turnOn instead')
  void activate() {
    turnOn();
  }

  void turnOn() {
    // ···
  }
}
```

### @deprecated
A general deprecation marker for unspecified future releases. The Dart team recommends using `@Deprecated` with explicit messaging instead.

### @override
Marks instance members as overrides or implementations of parent class or interface members.

### @pragma
Provides instructions or hints to Dart tools like compilers and analyzers about specific declarations.

## Analyzer-Supported Annotations

The Dart analyzer supports additional annotations from `package:meta`:

### @visibleForTesting
Marks package members as public exclusively for test access. The analyzer hides these from autocompletion and warns about external package usage.

### @awaitNotRequired
Indicates that Future-typed variables or Future-returning functions don't require callers to await them, suppressing relevant lint warnings.

## Custom Annotations

### Basic Definition

Create custom annotations as constant classes:

```dart
class Todo {
  final String who;
  final String what;

  const Todo(this.who, this.what);
}
```

### Usage

```dart
@Todo('Dash', 'Implement this function')
void doSomething() {
  print('Do something');
}
```

### Specifying Supported Targets

Use `@Target` from `package:meta` to restrict annotation application:

```dart
import 'package:meta/meta_meta.dart';

@Target({TargetKind.function, TargetKind.method})
class Todo {
  // ···
}
```

This configuration ensures the analyzer warns when the annotation is misapplied to unsupported constructs.
