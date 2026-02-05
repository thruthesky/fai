# Dart: extends vs with vs implements

## Overview

In Dart, three keywords enable code reuse and inheritance patterns:
- **extends**: Classical inheritance
- **implements**: Interface contracts
- **with**: Mixin composition

---

## extends - Inheritance

### Concept

The `extends` keyword allows a class to inherit properties and methods from a parent class. The child class automatically receives all non-overridden functionality.

### Key Points

- Creates an "is-a" relationship
- Child class inherits parent implementation
- Can override methods for specific behavior
- Only one parent class allowed (single inheritance)

### Example

```dart
class First {
    static int num = 1;
    void firstFunc() {
        print('hello');
    }
}

class Second extends First {
    // Inherits firstFunc() - no override needed
}

void main() {
    var second = Second();
    second.firstFunc();  // Output: hello
    print(First.num);    // Output: 1
}
```

---

## implements - Interfaces

### Concept

The `implements` keyword enforces a contract where the implementing class must define all methods from the interface. It's a "can-do" relationship rather than "is-a".

### Key Points

- Requires complete method redefinition
- No inherited implementation
- Enforces type contract
- A class can implement multiple interfaces

### Example

```dart
class First {
    void firstFunc() {
        print('hello');
    }
}

class Second implements First {
    @override
    void firstFunc() {
        print('We had to declare the methods of implemented class');
    }
}

void main() {
    var second = Second();
    second.firstFunc();
    // Output: We had to declare the methods of implemented class
}
```

---

## with - Mixins

### Concept

Mixins (via the `with` keyword) provide code reuse across multiple class hierarchies without multiple inheritance. Mixins are classes without constructors.

### Key Points

- Enables composition without inheritance chains
- Multiple mixins allowed (comma-separated)
- Methods can be overridden if needed
- "Shares functionality" pattern

### Example

```dart
mixin First {
    void firstFunc() {
        print('hello');
    }
}

mixin temp {
    void number() {
        print(10);
    }
}

class Second with First, temp {
    @override
    void firstFunc() {
        print('can override if needed');
    }
}

void main() {
    var second = Second();
    second.firstFunc();  // Output: can override if needed
    second.number();     // Output: 10
}
```

---

## Comparison Summary

| Feature | extends | implements | with |
|---------|---------|-----------|------|
| Relationship | is-a | can-do | uses |
| Implementation | Inherited | Must define | Inherited |
| Multiple allowed | No | Yes | Yes |
| Override required | Optional | Required | Optional |
| Use case | Base functionality | Contracts | Shared code |

---

## When to Use Each

- **extends**: Building class hierarchies with shared base behavior
- **implements**: Defining contracts that unrelated classes must follow
- **with**: Reusing specific functionality across multiple unrelated classes

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-extends-vs-with-vs-implements/
- **Fetched**: 2026-01-27
