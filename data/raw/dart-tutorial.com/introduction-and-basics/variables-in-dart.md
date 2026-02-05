# Variables in Dart

## Variables

Variables function as containers that hold values during program execution. Different variable types accommodate various value kinds. Basic example:

```dart
var name = "John";
```

## Variable Types

Dart supports several data types:

- **String**: Text values (e.g., "John") - requires quotes
- **int**: Integer numbers (e.g., 10, -10, 8555) - no decimals
- **double**: Floating-point numbers (e.g., 10.0, -10.2, 85.698) - includes decimals
- **num**: Any numeric type (e.g., 10, 20.2, -20) - accepts both int and double
- **bool**: Boolean values (true or false only)
- **var**: Any value type (e.g., 'Bimal', 12, 'z', true)

## Syntax

```
type variableName = value;
```

## Example: Using Variables in Dart

```dart
void main() {
  String name = "John";
  String address = "USA";
  num age = 20;
  num height = 5.9;
  bool isMarried = false;

  print("Name is $name");
  print("Address is $address");
  print("Age is $age");
  print("Height is $height");
  print("Married Status is $isMarried");
}
```

**Output:**
```
Name is John
Address is USA
Age is 20
Height is 5.9
Married Status is false
```

## Rules for Creating Variables

- Names are case-sensitive (a ≠ A)
- Names may contain letters and numbers
- Names cannot begin with numbers
- Keywords cannot be used as names
- Spaces are not permitted
- Special characters forbidden except underscore (_) and dollar sign ($)

## Dart Constants

Constants represent values that remain unchangeable after declaration. Mutable values are changeable; immutable values are not. Use the `const` keyword:

```dart
void main(){
  const pi = 3.14;
  pi = 4.23; // Error: not possible
  print("Value of PI is $pi");
}
```

## Naming Convention

Follow lowerCamelCase convention: start with lowercase, capitalize subsequent word beginnings (e.g., num1, fullName, isMarried).

**Recommended:**
```dart
var fullName = "John Doe";
const pi = 3.14;
```

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/variables-in-dart/
- **Fetched**: 2026-01-27
