# Class Modifiers in Dart

## Overview

"Modifier keywords for class declarations to control external library access." Class modifiers regulate how classes and mixins can be used both within their own library and externally. These require Dart 3.0+ (except `abstract`).

## Available Modifiers

### No Modifier (Unrestricted)

A class without modifiers allows:
- Constructing new instances
- Extending the class
- Implementing the interface
- Mixing in the class or mixin

### Abstract

"To define a class that doesn't require a full, concrete implementation of its entire interface, use the `abstract` modifier." Abstract classes cannot be instantiated from any library but can be extended or implemented. They often contain abstract methods.

```dart
abstract class Vehicle {
  void moveForward(int meters);
}
```

### Base

The `base` modifier enforces implementation inheritance within the library. "A base class disallows implementation outside of its own library." Key guarantees include:
- Base class constructors execute when subtypes are instantiated
- All private members exist in subtypes
- New members don't break existing subtypes

Classes extending or implementing `base` classes must themselves be marked `base`, `final`, or `sealed`.

```dart
base class Vehicle {
  void moveForward(int meters) { }
}
```

### Interface

"To define an interface, use the `interface` modifier. Libraries outside of the interface's own defining library can implement the interface, but not extend it." This prevents the fragile base class problem by limiting method override possibilities.

```dart
interface class Vehicle {
  void moveForward(int meters) { }
}
```

#### Abstract Interface

Combining `abstract` and `interface` creates a pure interface that can be implemented but not inherited by external libraries, and can contain abstract members.

### Final

"To close the type hierarchy, use the `final` modifier." Final classes prevent both inheritance and implementation outside their library. Subclasses within the library must be marked `base`, `final`, or `sealed`.

```dart
final class Vehicle {
  void moveForward(int meters) { }
}
```

### Sealed

"To create a known, enumerable set of subtypes, use the `sealed` modifier." Sealed classes are implicitly abstract and enable exhaustive switch statements. Only direct subtypes defined in the same library are possible, supporting compile-time exhaustiveness checking.

```dart
sealed class Vehicle {}
class Car extends Vehicle {}
class Truck implements Vehicle {}
```

## Combining Modifiers

Modifiers can be layered in this order:
1. Optional `abstract`
2. One of `base`, `interface`, `final`, or `sealed`
3. Optional `mixin`
4. `class` keyword

Incompatible combinations include `abstract` with `sealed` (redundant), and `interface`/`final`/`sealed` with `mixin` (prevent mixing).
