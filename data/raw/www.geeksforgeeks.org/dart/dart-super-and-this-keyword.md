# Dart - Super and This Keywords Tutorial

## Super Keyword in Dart

### Overview
The `super` keyword refers to the immediate parent class object. It enables access to parent class properties and methods. When a subclass instance is created, the parent class instance is implicitly instantiated first.

### Key Advantages
- Access parent class data members when both parent and child share the same member name
- Prevent unintended method overriding
- Call parameterized constructors from the parent class

### Syntax
```dart
super.variable_name;      // Access parent variables
super.method_name();       // Call parent methods
```

### Example 1: Constructor Flow in Inheritance
```dart
class SuperGeek {
  SuperGeek() {
    print("You are inside the Parent constructor!!");
  }
}

class SubGeek extends SuperGeek {
  SubGeek() {
    print("You are inside the Child constructor!!");
  }
}

void main() {
  SubGeek geek = SubGeek();
}
```

**Output:**
```
You are inside the Parent constructor!!
You are inside the Child constructor!!
```

The parent constructor executes before the child constructor during instantiation.

### Example 2: Accessing Parent Class Variables
```dart
class SuperGeek {
  String geek = "Geeks for Geeks";
}

class SubGeek extends SuperGeek {
  void printInfo() {
    print(super.geek);
  }
}

void main() {
  SubGeek geek = SubGeek();
  geek.printInfo();
}
```

**Output:**
```
Geeks for Geeks
```

### Example 3: Accessing Parent Class Methods
```dart
class SuperGeek {
  void printInfo() {
    print("Welcome to Gfg!!\nYou are inside the parent class.");
  }
}

class SubGeek extends SuperGeek {
  void info() {
    print("You are calling a method of the parent class.");
    super.printInfo();
  }
}

void main() {
  SubGeek geek = SubGeek();
  geek.info();
}
```

**Output:**
```
You are calling method of parent class.
Welcome to Gfg!!
You are inside parent class.
```

---

## This Keyword in Dart

### Overview
While `super` references the parent class, `this` refers to the current class instance itself.

### Key Advantages
- Reference the current class instance
- Assign values to instance variables
- Return the current class instance

### Example: Using This Keyword
```dart
class Geek {
  String geek_info = "";

  Geek(String info) {
    this.geek_info = info;
  }

  void printInfo() {
    print("Welcome to $geek_info");
  }
}

void main() {
  Geek geek = Geek("Geeks for Geeks");
  geek.printInfo();
}
```

**Output:**
```
Welcome to Geeks for Geeks
```

---

## Conclusion

- Deploy `super` to access parent class variables, methods, and constructors
- Use `this` to reference current class instance variables and methods
- Both keywords strengthen class relationship structure and code clarity in Dart

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-super-and-this-keyword/
- **Fetched**: 2026-01-27
