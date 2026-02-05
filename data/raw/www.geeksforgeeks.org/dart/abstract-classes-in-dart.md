# Abstract Classes in Dart

## Definition

An abstract class in Dart is defined as a class that contains one or more abstract methods (methods without implementation). To declare one, use the `abstract` keyword.

## Key Features

- **Abstract Methods**: Classes containing abstract methods must be declared abstract and may include both abstract and concrete methods
- **Declaration**: Use the `abstract` keyword to declare an abstract class
- **Initialization**: Abstract classes cannot be instantiated
- **Inheritance**: Subclasses extending an abstract class must implement all abstract methods

## Syntax

```dart
abstract class ClassName {
  // Body of the abstract class
}
```

## Code Example: Single Method Override

```dart
abstract class Gfg {
  void say();
  void write();
}

class Geeksforgeeks extends Gfg {
  @override
  void say() {
    print("Yo Geek!!");
  }

  @override
  void write() {
    print("Geeks For Geeks");
  }
}

void main() {
  Geeksforgeeks geek = Geeksforgeeks();
  geek.say();
  geek.write();
}
```

**Output:**
```
Yo Geek!!
Geeks For Geeks
```

## Code Example: Multiple Classes Override

```dart
abstract class Gfg {
  void geek_info();
}

class Geek1 extends Gfg {
  @override
  void geek_info() {
    print("This is Class Geek1.");
  }
}

class Geek2 extends Gfg {
  @override
  void geek_info() {
    print("This is Class Geek2.");
  }
}

void main() {
  Geek1 g1 = Geek1();
  g1.geek_info();
  Geek2 g2 = Geek2();
  g2.geek_info();
}
```

**Output:**
```
This is Class Geek1.
This is Class Geek2.
```

## Important Notes

- The `@override` annotation increases code readability and helps avoid errors in large codebases
- The `new` keyword is optional in modern Dart—instances are created automatically
- While Dart implicitly overrides methods, using `@override` provides explicit indication and prevents errors

## Conclusion

Abstract classes establish foundational structures in object-oriented programming by enforcing required implementations in subclasses, promoting code organization, reusability, and maintainability in Dart applications.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/abstract-classes-in-dart/
- **Fetched**: 2026-01-27
