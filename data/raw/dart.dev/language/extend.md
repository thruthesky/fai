# Extend a Class in Dart

## Overview

Dart allows you to create subclasses from a superclass using the `extends` keyword. You can reference the parent class using `super`.

## Basic Syntax

```dart
class Television {
  void turnOn() {
    _illuminateDisplay();
    _activateIrSensor();
  }
}

class SmartTelevision extends Television {
  void turnOn() {
    super.turnOn();
    _bootNetworkInterface();
    _initializeMemory();
    _upgradeApps();
  }
}
```

The child class calls the parent's implementation using `super.turnOn()` before adding its own functionality.

## Overriding Members

Subclasses can override instance methods, operators, getters, and setters. Use the `@override` annotation to clearly indicate intentional overrides:

```dart
class Television {
  set contrast(int value) {
    // ···
  }
}

class SmartTelevision extends Television {
  @override
  set contrast(num value) {
    // ···
  }
}
```

### Override Rules

When overriding methods, follow these requirements:

- **Return types**: Must match or be a subtype of the parent's return type
- **Parameter types**: Must match or be a supertype of the parent's types
- **Positional parameters**: Must accept the same number as the parent
- **Generic methods**: Cannot override non-generic methods, and vice versa

**Important**: Always override `hashCode` when overriding `==` to maintain consistency.

## noSuchMethod()

Override `noSuchMethod()` to handle calls to non-existent methods or variables:

```dart
class A {
  @override
  void noSuchMethod(Invocation invocation) {
    print(
      'You tried to use a non-existent member: '
      '${invocation.memberName}',
    );
  }
}
```

You can only invoke unimplemented methods if the receiver has dynamic typing or if it defines the method with a custom `noSuchMethod()` implementation.
