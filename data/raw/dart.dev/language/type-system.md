# The Dart Type System

## Overview

Dart employs a type-safe approach combining static type checking with runtime verification to guarantee variables match their declared types—a concept known as sound typing. While type annotations are technically optional due to type inference capabilities, the language encourages their use for better code quality and compile-time error detection.

## What is Soundness?

Soundness ensures programs cannot enter invalid states by guaranteeing "an expression evaluates to a value that doesn't mismatch the expression's static type." Dart's type system enforces this through compile-time analysis and runtime validation. For instance, assigning a string to an integer variable triggers a compile error, while unsafe casting (using `as`) may fail during execution.

## Benefits of Sound Typing

- **Early bug detection**: Type-related issues surface during compilation rather than runtime
- **Enhanced readability**: Code clarity improves when types are reliable and unambiguous
- **Simplified maintenance**: Changes propagate with type-system warnings preventing breakage
- **Efficient compilation**: AOT compilation generates substantially better code with type information

## Static Analysis Best Practices

### Return Type Covariance

Override methods should use return types matching or narrowing the parent's type. A subclass getter returning `HoneyBadger` instead of `Animal` remains valid; unrelated types violate type safety.

### Parameter Type Contravariance

Parameter types in overridden methods must match or broaden (use supertypes of) the original. Narrowing parameters breaks type safety—a `Cat` subclass accepting only `Mouse` arguments would fail if treated as a generic `Animal`.

### Dynamic List Restrictions

Assigning dynamic lists to typed lists creates unsound situations. `List<dynamic>` cannot substitute for `List<int>` without explicit casting, which may fail at runtime.

## Runtime Validation

Runtime checks address compile-time undetectable issues. Casting a `List<Dog>` as `List<Cat>` throws exceptions when the actual type doesn't match the declared target.

## Type Inference Mechanisms

The analyzer automatically derives types for:
- **Fields and methods**: Inherit superclass types or infer from initializers
- **Static variables**: Infer from initialization expressions
- **Local variables**: Derive from initial assignment only
- **Generic arguments**: Combine contextual information with argument types

### Inference Using Bounds

With language version 3.7.0+, the inference algorithm generates constraints by merging existing constraints with declared type bounds. This particularly benefits F-bounded generics, allowing expressions like `max(3, 7)` to correctly infer `max<num>` without explicit type parameters.

## Type Substitution Rules

**Core principle**: Replace consumer types with supertypes; replace producer types with subtypes.

### Simple Assignment Examples

- `Animal cat = Cat()` ✓ (consumer accepts supertype)
- `MaineCoon cat = Cat()` ✗ (consumer rejects subtype)
- `Cat cat = MaineCoon()` ✓ (producer provides subtype)

### Generic Assignment

`List<MaineCoon>` substitutes for `List<Cat>`, but `List<Animal>` cannot replace `List<Cat>` without explicit casting (which may fail at runtime).

### Method Overrides

Methods follow identical rules—parameters accept supertypes, returns accept subtypes.

## Covariant Parameters

The `covariant` keyword allows intentional parameter type narrowing in overrides, deferring validation to runtime. This rarely-used pattern trades compile-time safety for runtime checks:

```dart
class Cat extends Animal {
  @override
  void chase(covariant Mouse x) { ... }
}
```

## Code Example: Type Safety in Practice

```dart
void printInts(List<int> a) => print(a);

void main() {
  final list = <int>[]; // Explicit type annotation
  list.add(1);
  list.add(2);
  printInts(list);
}
```

Without the `<int>` annotation, the analyzer infers `List<dynamic>`, triggering a mismatch error when passed to a function expecting `List<int>`.

## Implicit Downcasts from Dynamic

Expressions typed as `dynamic` implicitly cast to specific types, with runtime validation. This can mask errors—enabling strict casts mode in analysis options prevents problematic implicit conversions.

## Additional Resources

- Type promotion troubleshooting guides
- Sound null safety documentation
- Custom analyzer configuration documentation
