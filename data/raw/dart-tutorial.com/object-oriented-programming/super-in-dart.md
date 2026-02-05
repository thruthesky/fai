# Super in Dart

## Definition

"Super is used to refer to the parent class. It is used to call the parent class's properties and methods."

## Key Uses

The `super` keyword enables developers to:
- Access parent class methods
- Reference parent class properties
- Call parent class constructors

## Code Examples

### Example 1: Calling Parent Methods

```dart
class Laptop {
  void show() {
    print("Laptop show method");
  }
}

class MacBook extends Laptop {
  void show() {
    super.show(); // Calling the show method of the parent class
    print("MacBook show method");
  }
}

void main() {
  MacBook macbook = MacBook();
  macbook.show();
}
```

**Output:**
```
Laptop show method
MacBook show method
```

### Example 2: Accessing Parent Properties

```dart
class Car {
  int noOfSeats = 4;
}

class Tesla extends Car {
  int noOfSeats = 6;

  void display() {
    print("No of seats in Tesla: $noOfSeats");
    print("No of seats in Car: ${super.noOfSeats}");
  }
}

void main() {
  var tesla = Tesla();
  tesla.display();
}
```

**Output:**
```
No of seats in Tesla: 6
No of seats in Car: 4
```

### Example 3: Using Super with Constructors

```dart
class Employee {
  Employee(String name, double salary) {
    print("Employee constructor");
    print("Name: $name");
    print("Salary: $salary");
  }
}

class Manager extends Employee {
  Manager(String name, double salary) : super(name, salary) {
    print("Manager constructor");
  }
}

void main() {
  Manager manager = Manager("John", 25000.0);
}
```

**Output:**
```
Employee constructor
Name: John
Salary: 25000.0
Manager constructor
```

### Example 4: Super with Named Constructors

```dart
class Employee {
  Employee.manager() {
    print("Employee named constructor");
  }
}

class Manager extends Employee {
  Manager.manager() : super.manager() {
    print("Manager named constructor");
  }
}

void main() {
  Manager manager = Manager.manager();
}
```

**Output:**
```
Employee named constructor
Manager named constructor
```

### Example 5: Multilevel Inheritance

```dart
class Laptop {
  void display() {
    print("Laptop display");
  }
}

class MacBook extends Laptop {
  void display() {
    print("MacBook display");
    super.display();
  }
}

class MacBookPro extends MacBook {
  void display() {
    print("MacBookPro display");
    super.display();
  }
}

void main() {
  var macbookpro = MacBookPro();
  macbookpro.display();
}
```

**Output:**
```
MacBookPro display
MacBook display
Laptop display
```

## Key Points

- "The super keyword is used to access the parent class members"
- "The super keyword is used to call the method of the parent class"
- Works with regular methods, properties, and constructors
- Supports named constructors
- Enables proper method chaining in multilevel inheritance hierarchies

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/super-in-dart/
- **Fetched**: 2026-01-27
