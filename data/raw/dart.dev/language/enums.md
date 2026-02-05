# Enumerated Types in Dart

## Overview

"Enumerated types, often called _enumerations_ or _enums_, are a special kind of class used to represent a fixed number of constant values." All enums automatically extend the `Enum` class and are sealed, preventing subclassing or explicit instantiation.

## Declaring Simple Enums

The basic syntax uses the `enum` keyword followed by value names:

```dart
enum Color { red, green, blue }
```

Trailing commas are supported to help prevent copy-paste errors.

## Declaring Enhanced Enums

Enhanced enums function like classes with fields, methods, and const constructors. Requirements include:

- Instance variables must be `final`
- All generative constructors must be constant
- Factory constructors can only return fixed enum instances
- Cannot extend classes other than `Enum`
- Cannot override `index`, `hashCode`, or equality operators
- Member named `values` is prohibited
- All instances must be declared at the beginning

### Example Enhanced Enum

```dart
enum Vehicle implements Comparable<Vehicle> {
  car(tires: 4, passengers: 5, carbonPerKilometer: 400),
  bus(tires: 6, passengers: 50, carbonPerKilometer: 800),
  bicycle(tires: 2, passengers: 1, carbonPerKilometer: 0);

  const Vehicle({
    required this.tires,
    required this.passengers,
    required this.carbonPerKilometer,
  });

  final int tires;
  final int passengers;
  final int carbonPerKilometer;

  int get carbonFootprint => (carbonPerKilometer / passengers).round();

  bool get isTwoWheeled => this == Vehicle.bicycle;

  @override
  int compareTo(Vehicle other) => carbonFootprint - other.carbonFootprint;
}
```

Enhanced enums require language version 2.17 or later.

## Using Enums

### Accessing Values

```dart
final favoriteColor = Color.blue;
if (favoriteColor == Color.blue) {
  print('Your favorite color is blue!');
}
```

### Index Property

Each value has a zero-based `index` getter:

```dart
assert(Color.red.index == 0);
assert(Color.green.index == 1);
assert(Color.blue.index == 2);
```

### Getting All Values

```dart
List<Color> colors = Color.values;
assert(colors[2] == Color.blue);
```

### Switch Statements

```dart
var aColor = Color.blue;

switch (aColor) {
  case Color.red:
    print('Red as roses!');
  case Color.green:
    print('Green as grass!');
  default:
    print(aColor);
}
```

The compiler warns if all enum values aren't handled.

### Name Property

Access the string name of an enum value:

```dart
print(Color.blue.name); // 'blue'
```

### Accessing Members

Members of enhanced enums are accessed like normal object properties:

```dart
print(Vehicle.car.carbonFootprint);
```
