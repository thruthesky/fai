# Mixins in Dart

## Introduction

Mixins are a mechanism for reusing code across multiple classes. They follow the DRY (Don't Repeat Yourself) principle and utilize three keywords: **mixin**, **with**, and **on**.

## Rules for Mixins

- Cannot be instantiated (no object creation)
- Enable code sharing between multiple classes
- Do not have constructors and cannot be extended
- Multiple mixins can be used in a single class

## Syntax

```dart
mixin Mixin1{
  // code
}

mixin Mixin2{
  // code
}

class ClassName with Mixin1, Mixin2{
  // code
}
```

## Example 1: Basic Mixin Usage

Two mixins provide methods to a class:

```dart
mixin ElectricVariant {
  void electricVariant() {
    print('This is an electric variant');
  }
}

mixin PetrolVariant {
  void petrolVariant() {
    print('This is a petrol variant');
  }
}

class Car with ElectricVariant, PetrolVariant {
  // access to both mixin methods
}

void main() {
  var car = Car();
  car.electricVariant();
  car.petrolVariant();
}
```

**Output:**
```
This is an electric variant
This is a petrol variant
```

## Example 2: Multiple Classes Using Different Mixins

```dart
mixin CanFly {
  void fly() {
    print('I can fly');
  }
}

mixin CanWalk {
  void walk() {
    print('I can walk');
  }
}

class Bird with CanFly, CanWalk {}
class Human with CanWalk {}

void main() {
  var bird = Bird();
  bird.fly();
  bird.walk();

  var human = Human();
  human.walk();
}
```

**Output:**
```
I can fly
I can walk
I can walk
```

## The "on" Keyword

Restricts mixin usage to specific classes.

### Syntax

```dart
mixin Mixin1 on Class1{
  // code
}
```

## Example 3: Using "on" Keyword

```dart
abstract class Animal {
  String name;
  double speed;

  Animal(this.name, this.speed);

  void run();
}

mixin CanRun on Animal {
  @override
  void run() => print('$name is Running at speed $speed');
}

class Dog extends Animal with CanRun {
  Dog(String name, double speed) : super(name, speed);
}

void main() {
  var dog = Dog('My Dog', 25);
  dog.run();
}
```

**Output:**
```
My Dog is Running at speed 25.0
```

## What Is Allowed

- Properties and static variables
- Regular, abstract, and static methods
- Multiple mixins per class

## What Is Not Allowed

- Constructor definitions
- Extending a mixin
- Creating mixin objects

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/mixins-in-dart/
- **Fetched**: 2026-01-27
