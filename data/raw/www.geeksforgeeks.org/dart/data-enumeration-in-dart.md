# Data Enumeration in Dart - Complete Tutorial

## Introduction

Enumerated types, commonly called enums, serve as a mechanism for defining named constant values in Dart. The `enum` keyword enables developers to create enumeration types that group related constant values under a single type definition.

## Syntax for Declaring Enums

```dart
enum variable_name {
    member1, member2, member3, ..., memberN
}
```

## Key Characteristics

**Important points about Dart enums:**

- The `enum` keyword initializes an enumerated data type
- Enums represent named constants rather than traditional classes
- Starting with Dart 2.17, enums gained support for fields, methods, and interface implementation
- Members must be separated by commas
- Unlike some languages, Dart does not automatically assign numeric values; instead, each enum value is an instance of its enum type
- No semicolon or comma should follow the final member

## Example 1: Printing All Enum Elements

```dart
// Creating enum with name Gfg
enum Gfg {
    Welcome,
    to,
    GeeksForGeeks,
}

void main() {
    // Printing values from the enum
    for (Gfg geek in Gfg.values) {
        print(geek);
    }
}
```

**Output:**
```
Gfg.Welcome
Gfg.to
Gfg.GeeksForGeeks
```

**Note:** Enum values display as `EnumType.valueName` because Dart invokes their `toString()` method automatically.

## Example 2: Using Switch-Case with Enums

```dart
enum Gfg { Welcome, to, GeeksForGeeks }

void main() {
    var geek = Gfg.GeeksForGeeks;

    switch (geek) {
        case Gfg.Welcome:
            print("This is not the correct case.");
            break;
        case Gfg.to:
            print("This is not the correct case.");
            break;
        case Gfg.GeeksForGeeks:
            print("This is the correct case.");
            break;
    }
}
```

**Output:**
```
This is the correct case.
```

## Limitations

- Enums cannot be subclassed or mixed in
- Explicit instantiation of enums is prohibited

## Summary

Dart's enumeration system provides a structured approach to managing finite sets of named constants. Unlike traditional integer-based implementations in other languages, Dart enums function as actual type instances. The evolution in Dart 2.17 introduced support for properties, methods, and interface adoption, expanding their utility in state administration and type categorization while maintaining their core restrictions.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/data-enumeration-in-dart/
- **Fetched**: 2026-01-27
