# Loops in Dart

## Dart Loops

In programming, loops execute a block of code repeatedly until certain conditions are met. Rather than manually repeating code many times, loops automate this process efficiently.

Dart supports four main loop types:

- **For Loop**
- **For Each Loop**
- **While Loop**
- **Do While Loop**

> "The primary purpose of all loops is to repeat a block of code."

## Print Your Name 10 Times Without Using Loop

Without loops, printing a name multiple times requires repetitive code:

```dart
void main() {
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
    print("John Doe");
}
```

**Output:**
```
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
```

## Print Your Name 10 Times Using Loop

Using a loop dramatically simplifies this task:

```dart
void main() {
  for (int i = 0; i < 10; i++) {
    print("John Doe");
  }
}
```

**Output:**
```
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
John Doe
```

For larger repetitions (like 1000 times), loops become essential.

> "Loops are helpful because they reduce errors, save time, and make code more readable."

## What After This?

Explore detailed sections on each loop type for comprehensive understanding.

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/loops-in-dart/
- **Fetched**: 2026-01-27
