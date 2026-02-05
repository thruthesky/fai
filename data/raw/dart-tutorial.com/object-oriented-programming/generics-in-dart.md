# Generics in Dart

## Introduction

Generics enable creating classes and functions that operate with multiple data types. The `List` class exemplifies this—`List<int>`, `List<String>`, and `List<double>` each work with specific types.

## Syntax

```dart
class ClassName<T> {
  // code
}
```

## Without Generics (Problematic Approach)

Creating separate classes for each type leads to code duplication:

```dart
class IntData {
  int data;
  IntData(this.data);
}

class DoubleData {
  double data;
  DoubleData(this.data);
}

void main() {
  IntData intData = IntData(10);
  DoubleData doubleData = DoubleData(10.5);
  print("IntData: ${intData.data}");
  print("DoubleData: ${doubleData.data}");
}
```

## With Generics (Recommended)

A single generic class handles multiple types:

```dart
class Data<T> {
  T data;
  Data(this.data);
}

void main() {
  Data<int> intData = Data<int>(10);
  Data<double> doubleData = Data<double>(10.5);

  print("IntData: ${intData.data}");
  print("DoubleData: ${doubleData.data}");
}
```

## Type Variables

Common naming conventions for type parameters:

| Variable | Purpose |
|----------|---------|
| T | Type |
| E | Element |
| K | Key |
| V | Value |

## Generic Methods

Methods can also be parameterized:

```dart
T genericMethod<T>(T value) {
  return value;
}

void main() {
  print("Int: ${genericMethod<int>(10)}");
  print("Double: ${genericMethod<double>(10.5)}");
  print("String: ${genericMethod<String>("Hello")}");
}
```

## Multiple Type Parameters

```dart
T genericMethod<T, U>(T value1, U value2) {
  return value1;
}

void main() {
  print(genericMethod<int, String>(10, "Hello"));
  print(genericMethod<String, int>("Hello", 10));
}
```

## Restricting Types with `extends`

Limit generic types using bounded type parameters:

```dart
class Data<T extends num> {
  T data;
  Data(this.data);
}

void main() {
  Data<int> intData = Data<int>(10);
  Data<double> doubleData = Data<double>(10.5);
  // Data<String> stringData = Data<String>("Hello"); // Not allowed
}
```

## Generic Method with Restriction

```dart
double getAverage<T extends num>(T value1, T value2) {
  return (value1 + value2) / 2;
}

void main() {
  print("Average of int: ${getAverage<int>(10, 20)}");
  print("Average of double: ${getAverage<double>(10.5, 20.5)}");
}
```

## Complex Generic Example

```dart
abstract class Shape {
  double get area;
}

class Circle implements Shape {
  final double radius;
  Circle(this.radius);

  @override
  double get area => 3.14 * radius * radius;
}

class Rectangle implements Shape {
  final double width;
  final double height;
  Rectangle(this.width, this.height);

  @override
  double get area => width * height;
}

class Region<T extends Shape> {
  List<T> shapes;
  Region({required this.shapes});

  double get totalArea {
    double total = 0;
    shapes.forEach((shape) {
      total += shape.area;
    });
    return total;
  }
}

void main() {
  var circle = Circle(10);
  var rectangle = Rectangle(10, 20);
  var region = Region(shapes: [circle, rectangle]);
  print("Total Area of Region: ${region.totalArea}");
}
```

## Key Advantages

- **Type Safety**: Prevents type-related errors at compile time
- **Code Reusability**: Write once, use with many types

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/generics-in-dart/
- **Fetched**: 2026-01-27
