# Understanding Null Safety in Dart

## Overview

Null safety represents a major enhancement to Dart's type system, addressing the fundamental issue of null reference errors. The document explains that "the goal is to give you *control* and *insight* into where `null` can flow through your program."

## Core Principles

The design follows three key principles:

1. **Safe by default** - Code avoids null reference errors through static checking unless you explicitly opt into unsafe features
2. **Usable syntax** - The implementation maintains Dart's familiar feel without sacrificing clarity
3. **Sound guarantees** - Non-nullable types provide compiler-verified safety with potential runtime optimizations

## Type System Changes

### Non-Nullable by Default

In pre-null-safety Dart, the `Null` type was a subtype of all types. The new system reverses this:

- **Non-nullable types** cannot contain `null` and support full method access
- **Nullable types** (marked with `?`) explicitly permit `null` but restrict operations

```dart
// Non-nullable - must contain a value
String name = "Alice";

// Nullable - may be null
String? nickname;
```

### Type Hierarchy

The change creates two parallel type universes:

- Non-nullable side: Full method access, cannot contain `null`
- Nullable side: Limited operations (only `Object` methods), permits `null`

Values flow safely from non-nullable to nullable types, but not vice versa. The system removed implicit downcasts entirely to maintain soundness.

## Flow Analysis

Dart's enhanced flow analysis enables practical null safety through several mechanisms:

### Type Promotion on Null Checks

Variables are automatically promoted to non-nullable types after null checks:

```dart
String? maybeString;
if (maybeString != null) {
  print(maybeString.length); // Safely promoted to String
}
```

### Reachability Analysis

The analyzer tracks control flow to understand unreachable code:

```dart
bool isEmptyList(Object object) {
  if (object is! List) return false;
  return object.isEmpty; // Object promoted to List
}
```

### The Never Type

The new bottom type `Never` indicates expressions that never complete normally:

```dart
Never throwError(String message) {
  throw Exception(message);
}

void validate(int value) {
  if (value < 0) throwError('Negative');
  // Compiler knows code after throwError cannot execute
}
```

## Working with Nullable Types

### Not-Null Assertion Operator

The `!` operator casts a nullable type to non-nullable, throwing at runtime if the value is null:

```dart
String? value = "text";
print(value!.length); // Assert non-null
```

### Late Variables

The `late` modifier defers initialization checks to runtime, enabling patterns like lazy initialization:

```dart
class Configuration {
  late String apiKey;

  void initialize() {
    apiKey = getKeyFromEnvironment();
  }
}
```

Combining `late` with `final` allows single assignment after declaration:

```dart
class User {
  late final String id;

  User(String userId) {
    id = userId; // Only assignment allowed
  }
}
```

### Null-Aware Operators

Enhanced null-aware operators short-circuit method chains:

```dart
// If person is null, entire chain stops
String? name = person?.getProfile()?.getName();

// Null-aware cascade
person?..updateStatus('away');

// Null-aware indexing
var value = map?[key];
```

### Required Named Parameters

Parameters can be required and named simultaneously:

```dart
void createUser({
  required String name,
  required int age,
  String? nickname,
}) {
  // name and age must be provided
  // nickname is optional
}
```

## Ensuring Correctness

### Variable Initialization Rules

- **Top-level and static variables** require explicit initializers
- **Instance fields** need initialization at declaration, in constructors, or initialization lists
- **Local variables** need initialization before use (verified by flow analysis)
- **Optional parameters** require either nullable types or default values

```dart
// Valid patterns
int x = 0; // Top-level initialized

class Example {
  int fieldA = 1; // Initialized at declaration
  int fieldB;

  Example(this.fieldB); // Initialized via formal parameter
}
```

### Return Value Guarantees

Non-nullable return types require every execution path to return a value. The analyzer intelligently handles control flow:

```dart
String getValue(bool condition) {
  if (condition) return "yes";
  return "no"; // All paths return something
}
```

## Generics and Nullability

Type parameters are "potentially nullable" - they may be instantiated as either nullable or non-nullable:

```dart
class Container<T> {
  final T? item; // Explicitly nullable to allow any type argument

  Container(this.item);
}

// Can instantiate with either nullable or non-nullable types
var stringBox = Container<String>("value");
var nullableIntBox = Container<int?>(null);
```

Bounded type parameters inherit nullability from their bounds:

```dart
class Range<T extends num> { // Non-nullable bound
  final T start, end;
  Range(this.start, this.end);
}

class Wrapper<T extends num?> { // Nullable bound
  final T value;
  Wrapper(this.value);
}
```

## Core Library Changes

### Map Index Operator

The `[]` operator returns nullable values since keys may not exist:

```dart
Map<String, int> data = {'count': 5};
int? value = data['count']; // Nullable return type
int actual = data['count']!; // Assert presence
```

### List Constructor Changes

The unnamed `List()` constructor is removed to prevent uninitialized elements. Use alternatives:

```dart
List<String> empty = List.empty();
List<int> filled = List.filled(5, 0);
List<double> generated = List.generate(5, (i) => i.toDouble());
```

### Length Setter Restrictions

Setting list length longer on non-nullable-typed lists throws an exception to prevent uninitialized element access. Truncating or growing nullable-typed lists remains allowed.

### Iterator Current Access

The `Iterator.current` property has non-nullable type. Accessing it outside valid iteration bounds produces undefined behavior (typically throws `StateError`).

## Practical Benefits

Sound null safety enables:

- **Compile-time error detection** - Catches potential null reference errors before runtime
- **Compiler optimizations** - Eliminates unnecessary null checks
- **Clearer code intent** - Explicit `?` and `!` operators show nullability handling
- **Gradual migration** - Mixed-version programs support legacy code alongside null-safe code

## Migration Path

Existing code can gradually adopt null safety. Within null-safe portions:

- Static type checker prevents null reference errors
- Runtime soundness applies to fully migrated code
- Mixed programs lose full soundness guarantees but gain static benefits in safe sections

The design intentionally preserves Dart's familiar patterns while adding safety guards, making adoption practical for existing codebases.
