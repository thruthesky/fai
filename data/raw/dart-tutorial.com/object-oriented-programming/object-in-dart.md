# Objects in Dart

## Definition

"An object is a self-contained unit of code and data. Objects are created from templates called classes. An object is made up of properties(variables) and methods(functions)." An object represents an instance of a class.

## Instantiation

Instantiation is the process of creating an instance from a class template. For example, if you define a Bicycle class, you can instantiate it as a bicycle object.

## Object Declaration Syntax

```dart
ClassName objectName = ClassName();
```

## Example 1: Bicycle Class

```dart
class Bicycle {
  String? color;
  int? size;
  int? currentSpeed;

  void changeGear(int newValue) {
    currentSpeed = newValue;
  }

  void display() {
    print("Color: $color");
    print("Size: $size");
    print("Current Speed: $currentSpeed");
  }
}

void main(){
    Bicycle bicycle = Bicycle();
    bicycle.color = "Red";
    bicycle.size = 26;
    bicycle.currentSpeed = 0;
    bicycle.changeGear(5);
    bicycle.display();
}
```

**Output:**
```
Color: Red
Size: 26
Current Speed: 5
```

## Example 2: Animal Class

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

## Example 3: Car Class

```dart
class Car {
  String? name;
  String? color;
  int? numberOfSeats;

  void start() {
    print("$name Car Started.");
  }
}

void main(){
    Car car = Car();
    car.name = "BMW";
    car.color = "Red";
    car.numberOfSeats = 4;
    car.start();

    Car car2 = Car();
    car2.name = "Audi";
    car2.color = "Black";
    car2.numberOfSeats = 4;
    car2.start();
}
```

**Output:**
```
BMW Car Started.
Audi Car Started.
```

## Accessing Object Members

"Once you create an object, you can access the properties and methods of the object using the dot(.) operator."

## Key Points

- The main() method serves as the program entry point
- The `new` keyword is optional when creating objects

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/object-in-dart/
- **Fetched**: 2026-01-27
