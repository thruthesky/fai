# Polymorphism in Dart

## Definition

"Poly means **many** and morph means **forms**. Polymorphism is the ability of an object to take on many forms."

In practical terms, polymorphism in object-oriented programming involves modifying or updating features and implementations that already exist in parent classes.

## Method Overriding

The primary technique for implementing polymorphism in Dart is method overriding—creating a method in a child class with the same name as a parent class method.

### Basic Syntax

```dart
class ParentClass {
  void functionName() {
  }
}

class ChildClass extends ParentClass {
  @override
  void functionName() {
  }
}
```

## Code Examples

### Example 1: Animal and Dog

```dart
class Animal {
  void eat() {
    print("Animal is eating");
  }
}

class Dog extends Animal {
  @override
  void eat() {
    print("Dog is eating");
  }
}

void main() {
  Animal animal = Animal();
  animal.eat();  // Output: Animal is eating

  Dog dog = Dog();
  dog.eat();     // Output: Dog is eating
}
```

### Example 2: Vehicle and Bus

```dart
class Vehicle {
  void run() {
    print("Vehicle is running");
  }
}

class Bus extends Vehicle {
  @override
  void run() {
    print("Bus is running");
  }
}
```

### Example 3: Multiple Inheritance

```dart
class Car {
  void power() {
    print("It runs on petrol.");
  }
}

class Honda extends Car {}

class Tesla extends Car {
  @override
  void power() {
    print("It runs on electricity.");
  }
}
```

### Example 4: Employee Hierarchy

```dart
class Employee {
  void salary() {
    print("Employee salary is \$1000.");
  }
}

class Manager extends Employee {
  @override
  void salary() {
    print("Manager salary is \$2000.");
  }
}

class Developer extends Employee {
  @override
  void salary() {
    print("Developer salary is \$3000.");
  }
}
```

## Key Benefits

- Subclasses can override parent class behavior
- Enables flexible and reusable code patterns

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/polymorphism-in-dart/
- **Fetched**: 2026-01-27
