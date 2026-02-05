# Basic Dart Program - Complete Content

## Overview

This tutorial covers foundational Dart programming concepts, starting with the classic "Hello World" program and progressing to practical examples.

## Basic Dart Program

The simplest Dart program prints output to the console:

```dart
void main() {
   print("Hello World!");
}
```

**Output:**
```
Hello World!
```

### Program Explanation

The tutorial notes that "void main() is the starting point where the execution of your program begins." Additional key points include:

- Every program requires a main function as its entry point
- Curly braces `{}` delimit code blocks
- The `print()` function outputs text to the screen
- All statements must end with semicolons

## Printing Variable Values

```dart
void main() {
    var name = "John";
    print(name);
}
```

**Output:**
```
John
```

## String Interpolation

To combine multiple variables, use the `$variableName` syntax:

```dart
void main(){
  var firstName = "John";
  var lastName = "Doe";
  print("Full name is $firstName $lastName");
}
```

**Output:**
```
Full name is John Doe
```

## Basic Calculations

```dart
void main() {
int num1 = 10;
int num2 = 3;

int sum = num1 + num2;
int diff = num1 - num2;
int mul = num1 * num2;
double div = num1 / num2;

print("The sum is $sum");
print("The diff is $diff");
print("The mul is $mul");
print("The div is $div");
}
```

**Output:**
```
The sum is 13
The diff is 7
The mul is 30
The div is 3.3333333333333335
```

## Creating a Dart Project

For larger applications, create a structured project using:

```bash
dart create <project_name>
```

### Project Setup Steps

1. Open command prompt/terminal in desired folder location
2. Execute `dart create project_name` (example: `dart create first_app`)
3. Navigate into project: `cd first_app`
4. Open in editor: `code .`
5. Access main file at `bin/first_app.dart`

## Running Projects

Execute your project with:

```bash
dart run
```

## Converting to JavaScript

| Command | Purpose |
|---------|---------|
| `dart compile js filename.dart` | Transpile Dart to JavaScript for Node.js execution |

## Challenge

Create a new Dart project named **stockmanagement** and execute it successfully.

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/basic-dart-program/
- **Fetched**: 2026-01-27
