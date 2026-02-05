# Getter in Dart

## Definition

A **getter** is used to retrieve the value of a property, primarily for accessing private properties. According to the tutorial, "Getter provide explicit read access to an object properties."

## Syntax

```dart
return_type get property_name {
  // Getter body
}
```

**Alternative syntax:** You can use the fat arrow (`=>`) instead of braces.

## Examples

### Example 1: Basic Getter

```dart
class Person {
  String? firstName;
  String? lastName;

  Person(this.firstName, this.lastName);

  String get fullName => "$firstName $lastName";
}

void main() {
  Person p = Person("John", "Doe");
  print(p.fullName);
}
```

Output: `John Doe`

### Example 2: Accessing Private Properties

```dart
class NoteBook {
  String? _name;
  double? _prize;

  NoteBook(this._name, this._prize);

  String get name => this._name!;
  double get price => this._prize!;
}

void main() {
  NoteBook nb = new NoteBook("Dell", 500);
  print(nb.name);
  print(nb.price);
}
```

### Example 3: Getter with Data Validation

```dart
class NoteBook {
  String _name;
  double _prize;

  NoteBook(this._name, this._prize);

  String get name {
    if (_name == "") {
      return "No Name";
    }
    return this._name;
  }

  double get price {
    return this._prize;
  }
}

void main() {
  NoteBook nb = new NoteBook("Apple", 1000);
  print("First Notebook name: ${nb.name}");
  print("First Notebook price: ${nb.price}");
  NoteBook nb2 = new NoteBook("", 500);
  print("Second Notebook name: ${nb2.name}");
  print("Second Notebook price: ${nb2.price}");
}
```

### Example 4: Multiple Getters and Map Getter

```dart
class Doctor {
  String _name;
  int _age;
  String _gender;

  Doctor(this._name, this._age, this._gender);

  String get name => _name;
  int get age => _age;
  String get gender => _gender;

  Map<String, dynamic> get map {
    return {"name": _name, "age": _age, "gender": _gender};
  }
}

void main() {
  Doctor d = Doctor("John", 41, "Male");
  print(d.map);
}
```

Output: `{name: John, age: 41, gender: Male}`

## Why Getters Matter

- Access private property values
- Restrict data member access within a class

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/getter-in-dart/
- **Fetched**: 2026-01-27
