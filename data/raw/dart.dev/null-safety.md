# Sound Null Safety in Dart

## Overview

Dart enforces sound null safety to prevent runtime errors from unintended `null` access. "Null safety prevents errors that result from unintentional access of variables set to `null`."

## Core Principles

Dart's null safety relies on two design principles:

**Non-nullable by default**: Variables cannot contain `null` unless explicitly marked. Research indicated non-null was the predominant choice in most APIs.

**Fully sound**: The type system guarantees non-nullable variables will never evaluate to `null` at runtime, enabling fewer bugs, smaller binaries, and faster execution.

## Type Annotations

With null safety enabled, standard variables require values:

```dart
var i = 42; // Inferred as int
String name = getFileName();
final b = Foo();
```

To allow `null` values, append `?` to the type:

```dart
int? aNullableInt = null;
```

## Error Detection

The Dart analyzer identifies potential null dereference issues at edit-time rather than runtime. It flags when non-nullable variables:

- Lack initialization with non-null values
- Are assigned `null`

## Dart 3 and Null Safety

Dart 3 includes built-in null safety and prevents execution of unsupported code. Migration requires updating the SDK constraint in `pubspec.yaml`:

```yaml
environment:
  sdk: '>=2.12.0 <3.0.0'
```

## Migration Path

For Dart 2.12–2.19, use the `dart migrate` tool:

```
dart migrate
```

Dart 3 removed this tool; consult the migration guide for manual updates.

## Resources

- Understanding null safety documentation
- Null safety migration guide
- Null safety FAQ
- Sample code repository
