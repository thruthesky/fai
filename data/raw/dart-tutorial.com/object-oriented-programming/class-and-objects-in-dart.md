# Classes and Objects in Dart

## What is a Class?

"A class is a blueprint for creating objects." Classes define the properties and methods that objects will possess.

## What is an Object?

"An object is an instance of a class." Multiple objects can be created from the same class definition.

## Example 1: Animal Class

This example demonstrates a basic class with properties and methods:

```dart
class Animal {
  String? name;
  int? numberOfLegs;
  int? lifeSpan;

  void display() {
    print("Animal name: $name.");
    print("Number of Legs: $numberOfLegs.");
    print("Life Span: $lifeSpan.");
  }
}

void main(){
  Animal animal = Animal();
  animal.name = "Lion";
  animal.numberOfLegs = 4;
  animal.lifeSpan = 10;
  animal.display();
}
```

**Output:**
```
Animal name: Lion.
Number of Legs: 4.
Life Span: 10.
```

## Example 2: Rectangle Area Calculator

This demonstrates calculating the area of a rectangle:

```dart
class Rectangle{
  double? length;
  double? breadth;

  double area(){
    return length! * breadth!;
  }
}

void main(){
  Rectangle rectangle = Rectangle();
  rectangle.length=10;
  rectangle.breadth=5;
  print("Area of rectangle is ${rectangle.area()}.");
}
```

**Output:**
```
Area of rectangle is 50.
```

## Example 3: Simple Interest Calculator

This calculates simple interest using class properties:

```dart
class SimpleInterest{
  double? principal;
  double? rate;
  double? time;

  double interest(){
    return (principal! * rate! * time!)/100;
  }
}

void main(){
  SimpleInterest simpleInterest = SimpleInterest();
  simpleInterest.principal=1000;
  simpleInterest.rate=10;
  simpleInterest.time=2;
  print("Simple Interest is ${simpleInterest.interest()}.");
}
```

**Output:**
```
Simple Interest is 200.
```

## Null Safety Note

"The **!** operator tells the compiler that the variable is not null," preventing errors in Dart's null safety system.

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/class-and-objects-in-dart/
- **Fetched**: 2026-01-27
