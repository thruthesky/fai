# Dart Comments - Complete Tutorial

## Overview

Comments serve as crucial documentation elements in programming. They are a set of statements that are not meant to be executed by the compiler and help developers understand code functionality both now and in the future.

## Types of Dart Comments

Dart supports three primary comment styles:

1. Single-line comments
2. Multi-line comments
3. Documentation comments

---

## 1. Single-Line Comments

Single-line comments use double forward slashes (`//`) and extend to the end of a line.

**Syntax:**
```dart
// This is a single line comment.
```

**Example:**
```dart
int main() {
    double area = 3.14 * 4 * 4;

    // It prints the area
    // of a circle of radius = 4
    print(area);

    return 0;
}
```

**Output:**
```
50.24
```

---

## 2. Multi-Line Comments

Multi-line comments use `/*` to begin and `*/` to end, allowing developers to comment entire code sections.

**Syntax:**
```dart
/*
These are
multiple line
of comments
*/
```

**Example:**
```dart
int main() {
    var lst = [1, 2, 3];

    /*
     It prints
     the whole list
     at once
    */
    print(lst);

    return 0;
}
```

**Output:**
```
[1, 2, 3]
```

---

## 3. Documentation Comments

Documentation comments provide references for packages, software, or projects. Dart supports two styles: `///` (C# style) and `/** ... */` (JavaDoc style). The `///` format is preferred because asterisks can create formatting conflicts in bulleted lists.

**Syntax:**
```dart
/// This is
/// a documentation
/// comment
```

**Example:**
```dart
bool checkEven(n) {
    /// Returns true
    /// if n is even
    if(n%2==0)
        return true;

    /// Returns false if n is odd
    else
        return false;
}

int main() {
    int n = 43;
    print(checkEven(n));
    return 0;
}
```

**Output:**
```
false
```

---

## Best Practices

Well-structured documentation comments enhance code quality and promote developer collaboration. They are particularly valuable when creating libraries, APIs, or large-scale applications, ensuring clarity and consistency throughout projects.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-comments/
- **Fetched**: 2026-01-27
