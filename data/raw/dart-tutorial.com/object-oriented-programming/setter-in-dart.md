# Setters in Dart

## Definition

"Setter is used to set the value of a property. It is mostly used to update a private property's value."

## Syntax

```dart
set property_name (value) {
  // Setter body
}
```

**Note:** You can use the arrow function syntax (`=>`) instead of braces.

## Example 1: Basic Setter

```dart
class NoteBook {
  String? _name;
  double? _prize;

  set name(String name) => this._name = name;
  set price(double price) => this._prize = price;

  void display() {
    print("Name: ${_name}");
    print("Price: ${_prize}");
  }
}

void main() {
  NoteBook nb = new NoteBook();
  nb.name = "Dell";
  nb.price = 500.00;
  nb.display();
}
```

**Output:**
```
Name: Dell
Price: 500.0
```

## Example 2: Setter with Data Validation

```dart
class NoteBook {
  String? _name;
  double? _prize;

  set name(String name) => _name = name;

  set price(double price) {
    if (price < 0) {
      throw Exception("Price cannot be less than 0");
    }
    this._prize = price;
  }

  void display() {
    print("Name: $_name");
    print("Price: $_prize");
  }
}

void main() {
  NoteBook nb = new NoteBook();
  nb.name = "Dell";
  nb.price = 250;
  nb.display();
}
```

## Example 3: Setter with Range Validation

```dart
class Student {
  String? _name;
  int? _classnumber;

  set name(String name) => this._name = name;

  set classnumber(int classnumber) {
    if (classnumber <= 0 || classnumber > 12) {
      throw ('Classnumber must be between 1 and 12');
    }
    this._classnumber = classnumber;
  }

  void display() {
    print("Name: $_name");
    print("Class Number: $_classnumber");
  }
}

void main() {
  Student s = new Student();
  s.name = "John Doe";
  s.classnumber = 12;
  s.display();
}
```

## Importance of Setters

- Enables updating private properties securely
- Implements data validation before assignment
- Provides better control over object data

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/setter-in-dart/
- **Fetched**: 2026-01-27
