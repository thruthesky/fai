# Typedefs in Dart

## Overview

A type alias, declared with the `typedef` keyword, provides a concise way to reference a type. This feature allows developers to create more readable and maintainable code by giving meaningful names to complex type declarations.

## Basic Type Aliases

The simplest form creates an alias for a concrete type:

```dart
typedef IntList = List<int>;
IntList il = [1, 2, 3];
```

This makes code clearer by replacing verbose generic syntax with a named reference.

## Generic Type Aliases

Type aliases support type parameters, enabling flexible, reusable type definitions:

```dart
typedef ListMapper<X> = Map<X, List<X>>;
Map<String, List<String>> m1 = {}; // Verbose
ListMapper<String> m2 = {}; // Clearer alternative
```

## Function Type Aliases

Function typedefs remain useful despite recommendations favoring inline function types in most situations:

```dart
typedef Compare<T> = int Function(T a, T b);

int sort(int a, int b) => a - b;

void main() {
  assert(sort is Compare<int>); // True!
}
```

## Language Version Requirements

As noted in the documentation, "the new typedefs requires a language version of at least 2.13." Prior versions restricted typedefs exclusively to function types.

## Key Benefits

Typedefs improve code readability by replacing lengthy generic declarations with meaningful names, making complex type signatures more accessible to developers reading the codebase.
