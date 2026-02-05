# Dart - Const and Final Keyword

## Overview

Dart supports assigning constant values to variables using two keywords that maintain immutability throughout the codebase:

- **const keyword**
- **final keyword**

These keywords are used to keep the value of a variable static throughout the code base, meaning once the variable is defined its state cannot be altered.

## Final Keyword in Dart

The `final` keyword hardcodes variable values that cannot be modified after initialization.

### Syntax

```dart
// Without datatype
final variable_name;

// With datatype
final data_type variable_name;
```

### Example

```dart
void main() {
  // Assigning value to geek1 without datatype
  final geek1 = "Geeks For Geeks";
  print(geek1);  // Output: Geeks For Geeks

  // Assigning value to geek2 with datatype
  final String geek2 = "Geeks For Geeks Again!!";
  print(geek2);  // Output: Geeks For Geeks Again!!
}
```

### Error Cases

**Uninitialized final variable:** Final variable 'geek1' must be assigned before it can be used.

**Reassignment attempt:** Can't assign to the final variable 'geek1'.

## Const Keyword in Dart

The `const` keyword creates compile-time constants. Using const on an object makes the object's entire deep state strictly fixed at compile-time, and the object with this state will be considered frozen and completely immutable.

### Syntax

```dart
const variable_name;
const data_type variable_name;
```

### Example

```dart
void main() {
  const geek1 = "Geeks For Geeks";
  print(geek1);  // Output: Geeks For Geeks

  const String geek2 = "Geeks For Geeks Again!!";
  print(geek2);  // Output: Geeks For Geeks Again!!
}
```

## Key Difference: Const vs Without Const

### Without Const

```dart
gfg() => [1, 2];

void main() {
  var geek1 = gfg();
  var geek2 = gfg();
  print(geek1 == geek2);  // Output: false
}
```

### With Const

```dart
gfg() => const[1, 2];

void main() {
  var geek1 = gfg();
  var geek2 = gfg();
  print(geek1 == geek2);  // Output: true
}
```

The const keyword enables canonicalization—identical const objects reference the same memory location.

## Const Keyword Properties

1. **Compile-time creation:** Data must exist during compilation; dynamic values like current time are invalid
2. **Deep immutability:** They are deeply and transitively immutable
3. **Canonicalization:** Identical const values share single instances

## Common Errors

**Uninitialized const:** The const variable 'geek1' must be initialized.

**Reassignment:** Can't assign to the const variable 'geek1'.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-const-and-final-keyword/
- **Fetched**: 2026-01-27
