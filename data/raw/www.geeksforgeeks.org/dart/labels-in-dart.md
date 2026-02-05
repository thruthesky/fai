# Labels in Dart - Tutorial

## Overview

Labels in Dart provide a way to control program flow in nested loops. Unlike C, Dart lacks `goto` statements but supports labels with `break` and `continue` for managing complex loop structures.

## Key Concept

Dart does allow line breaks between labels and loop control statements. Labels enable developers to exit or skip iterations in specific loops rather than just the innermost loop.

## Using Labels with Break

Labels used with `break` exit an entire labeled loop structure:

```dart
void main() {
  // Defining the label
  Geek1: for(int i = 0; i < 3; i++) {
    if(i < 2) {
      print("You are inside the loop Geek");
      // breaking with label
      break Geek1;
    }
    print("You are still inside the loop");
  }
}
```

**Output:**
```
You are inside the loop Geek
```

The loop terminates completely rather than continuing to subsequent iterations.

## Using Labels with Continue

Labels paired with `continue` skip the current iteration of a labeled loop:

```dart
void main() {
  // Defining the label
  Geek1: for(int i = 0; i < 3; i++) {
    if(i < 2) {
      print("You are inside the loop Geek");
      // Continue with label
      continue Geek1;
    }
    print("You are still inside the loop");
  }
}
```

**Output:**
```
You are inside the loop Geek
You are inside the loop Geek
You are still inside the loop
```

The loop continues rather than breaking, executing the labeled loop's next iteration.

## Key Differences

- **Break with label:** Terminates the entire labeled loop completely
- **Continue with label:** Skips only the current iteration, proceeding to the next one

## Practical Benefit

Labels simplify managing nested loops by allowing targeted control flow adjustments without unnecessary conditional nesting or redundant code structures.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/labels-in-dart/
- **Fetched**: 2026-01-27
