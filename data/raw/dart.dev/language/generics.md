# Generics in Dart

## Overview

Generics allow you to create reusable types with formal type parameters. The Dart `List` type, for example, is actually `List<E>`, where `E` represents a generic type variable. By convention, type variables use single-letter names like E, T, S, K, and V.

## Why Use Generics?

Generics serve two primary purposes:

**Type Safety and Better Code Generation**
Declaring specific types produces superior generated code. For instance, declaring `List<String>` enables tools to catch mistakes:

```dart
var names = <String>[];
names.addAll(['Seth', 'Kathy', 'Lars']);
names.add(42); // Error - type mismatch
```

**Reducing Code Duplication**
Instead of creating separate interfaces for each type you need, generics allow sharing a single interface. Rather than implementing `StringCache`, `IntCache`, and `ObjectCache` separately:

```dart
abstract class Cache<T> {
  T getByKey(String key);
  void setByKey(String key, T value);
}
```

This single definition handles all types.

## Collection Literals

Parameterize lists, sets, and maps using angle bracket notation:

```dart
var names = <String>['Seth', 'Kathy', 'Lars'];
var uniqueNames = <String>{'Seth', 'Kathy', 'Lars'};
var pages = <String, String>{
  'index.html': 'Homepage',
  'robots.txt': 'Hints for web robots',
};
```

## Parameterized Types with Constructors

Specify types in angle brackets after the class name:

```dart
var nameSet = Set<String>.of(names);
var views = SplayTreeMap<int, View>();
```

## Reified Generics

Unlike Java's type erasure, Dart's generics are *reified*—type information persists at runtime:

```dart
var names = <String>[];
print(names is List<String>); // true
```

## Restricting Parameterized Types

Use `extends` to limit what types can be provided. A common pattern ensures non-nullable types:

```dart
class Foo<T extends Object> {
  // T must be non-nullable
}
```

Restrict to specific base classes:

```dart
class Foo<T extends SomeBaseClass> {
  String toString() => "Instance of 'Foo<$T>'";
}

var someBaseClassFoo = Foo<SomeBaseClass>();
var extenderFoo = Foo<Extender>();
```

### F-Bounds (Self-Referential Constraints)

Type parameters can reference themselves, creating self-referential restrictions:

```dart
int compareAndOffset<T extends Comparable<T>>(T t1, T t2) =>
    t1.compareTo(t2) + 1;

class A implements Comparable<A> {
  @override
  int compareTo(A other) => 0;
}
```

The bound `T extends Comparable<T>` ensures T is comparable to itself.

## Generic Methods

Methods and functions support type arguments:

```dart
T first<T>(List<T> ts) {
  T tmp = ts[0];
  return tmp;
}
```

Here, `<T>` enables the type parameter in return types, argument types, and local variables.
