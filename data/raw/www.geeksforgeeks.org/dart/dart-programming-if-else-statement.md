# Dart Programming - If Else Statement Tutorial

## Overview

Decision-making statements allow programmers to determine which statements should be executed under different conditions. There are four primary types of decision-making structures in Dart.

## 1. If Statement

### Concept
This construct evaluates a condition and executes the enclosed code block only when the condition proves true.

### Syntax
```dart
if ( condition ){
    // body of if
}
```

### Example
```dart
void main() {
    int gfg = 10;

    // Condition is true
    if (gfg > 3) {
        // This will be printed
        print("Condition is true");
    }
}
```

**Output:**
```
Condition is true
```

---

## 2. If-Else Statement

### Concept
This statement checks a condition and executes one block if true, or an alternative block if false.

### Syntax
```dart
if ( condition ){
    // body of if
}
else {
    // body of else
}
```

### Example
```dart
void main() {
    int gfg = 10;

    // Condition is false
    if (gfg > 30) {
        // This will not be printed
        print("Condition is true");
    }
    else {
        // This will be printed
        print("Condition is false");
    }
}
```

**Output:**
```
Condition is false
```

---

## 3. Else-If Ladder

### Concept
This structure checks multiple conditions sequentially. This process is continued until the ladder is completed when a true condition executes its block.

### Syntax
```dart
if ( condition1 ){
    // body of if
}
else if ( condition2 ){
    // body of if
}
.
.
.
else {
    // statement
}
```

### Example
```dart
void main() {
    int gfg = 10;
    if (gfg < 9) {
        print("Condition 1 is true");
        gfg++;
    }
    else if (gfg < 10) {
        print("Condition 2 is true");
    }
    else if (gfg >= 10) {
        print("Condition 3 is true");
    }
    else if (++gfg > 11) {
        print("Condition 4 is true");
    }
    else {
        print("All the conditions are false");
    }
}
```

**Output:**
```
Condition 3 is true
```

---

## 4. Nested If Statement

### Concept
An if statement placed inside another if statement. The if statement inside it checks its condition and if true then the statements are executed.

### Syntax
```dart
if ( condition1 ){
    if ( condition2 ){
        // Body of if
    }
    else {
        // Body of else
    }
}
```

### Example
```dart
void main() {
    int gfg = 10;
    if (gfg > 9) {
        gfg++;
        if (gfg < 10) {
            print("Condition 2 is true");
        }
        else {
            print("All the conditions are false");
        }
    }
}
```

**Output:**
```
All the conditions are false
```

---

## Advanced Example with Operators

```dart
void main() {
    int x = 5, y = 7;

    // (++x > y--) -> x=6, y=7 (false), so (++x < ++y) is skipped
    if ((++x > y--) && (++x < ++y)) {
        print("Condition true");
    }
    else {
        // Executes since first condition is false
        print("Condition false");
    }

    // Final values after evaluation
    // 6 (incremented once)
    print(x);
    // 6 (decremented once)
    print(y);
}
```

**Output:**
```
Condition false
6
6
```

---

## Key Takeaways

- **if Statement:** Executes code only when a specific condition is true
- **if-else Statement:** Chooses between two execution paths based on condition evaluation
- **else-if Ladder:** Handles multiple conditions sequentially
- **Nested if Statement:** Manages complex conditional logic through hierarchical structures

These decision-making structures are essential for creating logical flows in programs, ensuring they execute efficiently and respond appropriately.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-programming-if-else-statement-if-if-else-nested-if-if-else-if/
- **Fetched**: 2026-01-27
