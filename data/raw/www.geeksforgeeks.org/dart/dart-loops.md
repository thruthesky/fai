# Dart Loops Tutorial

## Overview

A looping statement in Dart or any other programming language is used to repeat a particular set of commands until certain conditions are not completed.

Dart offers five main looping mechanisms:
- for loop
- for...in loop
- forEach loop
- while loop
- do...while loop

---

## For Loop

The traditional for loop works similarly to Java's implementation and executes code a predetermined number of times.

**Syntax:**
```dart
for(initialization; condition; increment/decrement expression){
  // Body of the loop
}
```

**Control Flow:**
1. **Initialization** - Sets up loop variables once at the start
2. **Condition** - Evaluates whether the loop should continue
3. **Body** - Contains code that executes repeatedly
4. **Increment/Update** - Modifies loop variables after each iteration

**Example:**
```dart
void main() {
  for (int i = 0; i < 5; i++) {
    print('GeeksForGeeks');
  }
}
```

**Output:**
```
GeeksForGeeks
GeeksForGeeks
GeeksForGeeks
GeeksForGeeks
GeeksForGeeks
```

---

## For...In Loop

This loop elegantly lets you iterate through the elements of a collection, such as lists or sets.

**Syntax:**
```dart
for (var element in collection) {
  // Body of loop
}
```

**Example:**
```dart
void main() {
  var numbers = [1, 2, 3, 4, 5];
  for (int i in numbers) {
    print(i);
  }
}
```

**Output:**
```
1
2
3
4
5
```

---

## ForEach Loop

A functional approach that applies an operation to each collection element.

**Syntax:**
```dart
collection.forEach((value) {
  // Body of loop
});
```

**Parameters:**
- `(value){}` defines a function specifying what action to perform on each item

**Example:**
```dart
void main() {
  var numbers = [1, 2, 3, 4, 5];
  numbers.forEach((num) => print(num));
}
```

**Output:**
```
1
2
3
4
5
```

---

## While Loop

The while loop keeps executing its block of code as long as the condition remains true.

**Syntax:**
```dart
while (condition) {
  // Body of loop
}
```

**Example:**
```dart
void main() {
  var limit = 4;
  int i = 1;
  while (i <= limit) {
    print('Hello Geek');
    i++;
  }
}
```

**Output:**
```
Hello Geek
Hello Geek
Hello Geek
Hello Geek
```

---

## Do...While Loop

The do...while ensures that its block of code runs at least once before it even bothers checking the condition.

**Syntax:**
```dart
do {
  // Body of loop
} while(condition);
```

**Example:**
```dart
void main() {
  var limit = 4;
  int i = 1;
  do {
    print('Hello Geek');
    i++;
  } while (i <= limit);
}
```

**Output:**
```
Hello Geek
Hello Geek
Hello Geek
Hello Geek
```

---

## Loop Selection Guide

- **for loop** - Use when iteration count is predetermined
- **for...in loop** - Simplifies iteration over lists and sets
- **forEach loop** - Offers functional programming style
- **while loop** - Useful when iteration count is uncertain
- **do...while loop** - Guarantees at least one execution

Selecting the appropriate loop depends on the specific problem, the data structure being used, and the requirements for control flow.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-loops/
- **Fetched**: 2026-01-27
