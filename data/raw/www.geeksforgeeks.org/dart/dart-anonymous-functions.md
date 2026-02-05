# Dart Anonymous Functions Tutorial

## Overview

An anonymous function in Dart is like a named function but they do not have names associated with it. These functions represent self-contained code blocks that can be passed as parameters throughout your application.

### Key Characteristics

- Nameless functions also known as lambdas or closures
- Can have zero or more parameters with optional type annotations
- Treated as first-class objects in Dart
- Can be assigned to variables or constants for later use

## Syntax

```dart
(parameter_list) {
    statement(s)
}
```

---

## Using Anonymous Functions with forEach()

### Example

```dart
void main() {
    var list = ["Shubham","Nick","Adil","Puthal"];
    print("GeeksforGeeks - Anonymous function in Dart");
    list.forEach((item) {
        print('${list.indexOf(item)} : $item');
    });
}
```

### Output
```
GeeksforGeeks - Anonymous function in Dart
0 : Shubham
1 : Nick
2 : Adil
3 : Puthal
```

The anonymous function accepts an untyped parameter and executes for each list item, printing both the index and value.

---

## Assigning Anonymous Functions to Variables

Since functions are first-class objects, you can store anonymous functions in variables for reuse:

### Example

```dart
void main() {
    var multiply = (int a, int b) {
        return a * b;
    };

    print(multiply(5, 3)); // Output: 15
}
```

### Output
```
15
```

---

## Anonymous Functions as Callbacks

Callbacks are particularly useful for asynchronous operations like event handlers:

### Example

```dart
void performOperation(int a, int b, Function operation) {
    print('Result: ${operation(a, b)}');
}

void main() {
    performOperation(4, 2, (a, b) => a + b);
}
```

### Output
```
Result: 6
```

This demonstrates passing an anonymous function directly as a parameter using arrow syntax `=>`.

---

## Key Takeaways

- Anonymous functions enable flexible, concise code patterns
- Useful for callbacks and functional programming approaches
- Can be assigned to variables for reusable operations
- Support optional type annotations for safety

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-anonymous-functions/
- **Fetched**: 2026-01-27
