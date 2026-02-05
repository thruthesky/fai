# Named Constructors in Dart

## Overview

Named constructors allow developers to create multiple constructors with different names in Dart. Unlike traditional languages (Java, C++, C#), Dart doesn't support constructor overloading, so this feature provides an alternative way to initialize objects with different parameters.

## Key Concept

"Named constructors improves code readability. It is useful when you want to create multiple constructors with the same name."

The syntax follows the pattern: `ClassName.constructorName(parameters)`

## Example 1: Basic Named Constructor

```dart
class Student {
  String? name;
  int? age;
  int? rollNumber;

  // Default Constructor
  Student() {
    print("This is a default constructor");
  }

  // Named Constructor
  Student.namedConstructor(String name, int age, int rollNumber) {
    this.name = name;
    this.age = age;
    this.rollNumber = rollNumber;
  }
}

void main() {
  Student student = Student.namedConstructor("John", 20, 1);
  print("Name: ${student.name}");
  print("Age: ${student.age}");
  print("Roll Number: ${student.rollNumber}");
}
```

**Output:**
```
This is a default constructor
Name: John
Age: 20
Roll Number: 1
```

## Example 2: Optional Parameters

```dart
class Mobile {
  String? name;
  String? color;
  int? price;

  Mobile(this.name, this.color, this.price);
  Mobile.namedConstructor(this.name, this.color, [this.price = 0]);

  void displayMobileDetails() {
    print("Mobile name: $name.");
    print("Mobile color: $color.");
    print("Mobile price: $price");
  }
}

void main() {
  var mobile1 = Mobile("Samsung", "Black", 20000);
  mobile1.displayMobileDetails();
  var mobile2 = Mobile.namedConstructor("Apple", "White");
  mobile2.displayMobileDetails();
}
```

**Output:**
```
Mobile name: Samsung.
Mobile color: Black.
Mobile price: 20000
Mobile name: Apple.
Mobile color: White.
Mobile price: 0
```

## Example 3: Multiple Named Constructors

```dart
class Animal {
  String? name;
  int? age;

  // Default Constructor
  Animal() {
    print("This is a default constructor");
  }

  // Named Constructor
  Animal.namedConstructor(String name, int age) {
    this.name = name;
    this.age = age;
  }

  // Another Named Constructor
  Animal.namedConstructor2(String name) {
    this.name = name;
  }
}

void main(){
  Animal animal = Animal.namedConstructor("Dog", 5);
  print("Name: ${animal.name}");
  print("Age: ${animal.age}");

  Animal animal2 = Animal.namedConstructor2("Cat");
  print("Name: ${animal2.name}");
}
```

**Output:**
```
Name: Dog
Age: 5
Name: Cat
```

## Example 4: Real-World Use Case - JSON Parsing

```dart
import 'dart:convert';

class Person {
  String? name;
  int? age;

  Person(this.name, this.age);

  Person.fromJson(Map<String, dynamic> json) {
    name = json['name'];
    age = json['age'];
  }

  Person.fromJsonString(String jsonString) {
    Map<String, dynamic> json = jsonDecode(jsonString);
    name = json['name'];
    age = json['age'];
  }
}

void main() {
  String jsonString1 = '{"name": "Bishworaj", "age": 25}';
  String jsonString2 = '{"name": "John", "age": 30}';

  Person p1 = Person.fromJsonString(jsonString1);
  print("Person 1 name: ${p1.name}");
  print("Person 1 age: ${p1.age}");

  Person p2 = Person.fromJsonString(jsonString2);
  print("Person 2 name: ${p2.name}");
  print("Person 2 age: ${p2.age}");
}
```

**Output:**
```
Person 1 name: Bishworaj
Person 1 age: 25
Person 2 name: John
Person 2 age: 30
```

## Challenge

Create a `Car` class with three properties (`name`, `color`, `price`), a `display()` method, a standard constructor accepting all parameters, and a named constructor accepting only `name` and `color`. Instantiate objects using both approaches and call `display()`.

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/named-constructor-in-dart/
- **Fetched**: 2026-01-27
