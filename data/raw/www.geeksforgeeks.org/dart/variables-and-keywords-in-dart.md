# Dart Variables Tutorial

## Overview

Variables represent named memory locations where data is stored. The specific type chosen depends on the data category being managed.

### Declaration Syntax

**Single variable:**
```
type variable_name;
```

**Multiple variables:**
```
type variable1_name, variable2_name, variable3_name;
```

## Variable Types

Dart supports three primary variable categories:
- Static variables
- Dynamic variables
- Final or const variables

## Naming Requirements

Variable identifiers must follow these rules:

1. Cannot use reserved keywords
2. May contain letters and numbers
3. Cannot include spaces or special characters (except underscore `_` and dollar `$`)
4. Cannot start with numeric characters

> **Important:** Dart supports type-checking, it means that it checks whether the data type and the data that variable holds are specific to that data or not.

## Code Examples

### Example 1: Basic Variable Declaration

```dart
void main() {
    int gfg1 = 10;
    double gfg2 = 0.2;
    bool gfg3 = false;
    String gfg4 = "0", gfg5 = "Geeks for Geeks";

    print(gfg1);  // 10
    print(gfg2);  // 0.2
    print(gfg3);  // false
    print(gfg4);  // 0
    print(gfg5);  // Geeks for Geeks
}
```

## Keywords

Reserved words cannot serve as variable identifiers since their functions are predefined in Dart.

## Dynamic Type Variables

The `dynamic` keyword allows variables to implicitly accept any value type during runtime, differing from `var` which becomes the assigned type permanently.

### Example 2: Dynamic Type Usage

```dart
void main() {
    dynamic geek = "Geeks For Geeks";
    print(geek);

    geek = 3.14157;
    print(geek);
}
```

**Output:**
```
Geeks For Geeks
3.14157
```

> Using `var` instead produces a compilation error when attempting this reassignment.

## Final and Const Keywords

These keywords define immutable variables—values cannot change after assignment.

### Final Variables

A final variable can only be set once and it is initialized when accessed.

**Syntax:**
```
final variable_name;
final data_type variable_name;
```

**Example:**
```dart
void main() {
    final geek1 = "Geeks For Geeks";
    print(geek1);

    final String geek2 = "Geeks For Geeks Again!!";
    print(geek2);
}
```

Attempting reassignment produces an error.

### Const Variables

Compile-time constants with values known before program execution. Must initialize during declaration.

**Syntax:**
```
const data_type variable_name;
```

**Example:**
```dart
void main() {
    const geek1 = "Geeks For Geeks";
    print(geek1);

    const geek2 = "Geeks For Geeks Again!!";
    print(geek2);
}
```

## Null Safety

By default, variables cannot hold null values without explicit declaration. To enable nullable types, append `?` to the type:

```dart
void main() {
    int? a;
    a = null;
    print(a);  // null
}
```

## Summary

Variables in Dart require adherence to specific naming conventions and type rules. The language offers flexibility through `dynamic` typing while enforcing immutability through `final` and `const` keywords. Null safety mechanisms require explicit opt-in for nullable types, enhancing code reliability.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/variables-and-keywords-in-dart/
- **Fetched**: 2026-01-27
