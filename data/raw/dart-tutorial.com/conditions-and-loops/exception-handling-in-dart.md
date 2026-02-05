# Exception Handling in Dart

## Exception In Dart

"An exception is an error that occurs at runtime during program execution. When the exception occurs, the flow of the program is interrupted, and the program terminates abnormally." Proper exception management prevents application crashes.

## Syntax

```dart
try {
  // Your Code Here
} catch(ex){
  // Exception here
}
```

## Try & Catch In Dart

**Try Block:** Contains code that might generate errors during execution.

**Catch Block:** Captures and handles exceptions when their types are uncertain.

## Example 1: Try Catch In Dart

This demonstrates basic exception handling when dividing by zero:

```dart
void main() {
   int a = 18;
   int b = 0;
   int res;

   try {
      res = a ~/ b;
      print("Result is $res");
   }
   catch(ex) {
      print(ex);
    }
}
```

**Output:** `IntegerDivisionByZeroException`

## Finally In Dart Try Catch

"The finally block is always executed whether the exceptions occur or not." It's optional but executes after try-catch blocks complete.

The **on** keyword targets specific exception types.

## Syntax

```dart
try {
  .....
}
on Exception1 {
  ....
}
catch Exception2 {
  ....
}
finally {
  // code that should always execute whether an exception or not.
}
```

## Example 2: Finally In Dart Try Catch

```dart
void main() {
  int a = 12;
  int b = 0;
  int res;
  try {
    res = a ~/ b;
  } on UnsupportedError {
    print('Cannot divide by zero');
  } catch (ex) {
    print(ex);
  } finally {
    print('Finally block always executed');
  }
}
```

**Output:**
```
Cannot divide by zero
Finally block always executed
```

## Throwing An Exception

The `throw` keyword explicitly raises exceptions, which must be handled to prevent unexpected program termination.

## Syntax

```dart
throw new Exception_name()
```

## Example 3: Throwing An Exception

```dart
void main() {
  try {
    check_account(-10);
  } catch (e) {
    print('The account cannot be negative');
  }
}

void check_account(int amount) {
  if (amount < 0) {
    throw new FormatException();
  }
}
```

**Output:** `The account cannot be negative`

## Why Is Exception Handling Needed?

Exception handling is essential for:

- Preventing abnormal program termination
- Avoiding logical errors during execution
- Maintaining application stability
- Reducing security vulnerabilities
- Providing better user experiences
- Enabling debugging and error resolution

## How To Create Custom Exception In Dart

Dart allows developers to create tailored exceptions for specific application needs.

## Syntax

```dart
class YourExceptionClass implements Exception{
  // constructors, variables & methods
}
```

## Example 4: How to Create & Handle Exception

```dart
class MarkException implements Exception {
  String errorMessage() {
    return 'Marks cannot be negative value.';
  }
}

void main() {
  try {
    checkMarks(-20);
  } catch (ex) {
    print(ex.toString());
  }
}

void checkMarks(int marks) {
  if (marks < 0) throw MarkException().errorMessage();
}
```

**Output:** `Marks cannot be negative value.`

## Example 5: How to Create & Handle Exception

```dart
import 'dart:math';

class NegativeSquareRootException implements Exception {
  @override
  String toString() {
    return 'Sqauare root of negative number is not allowed here.';
  }
}

num squareRoot(int i) {
  if (i < 0) {
    throw NegativeSquareRootException();
  } else {
    return sqrt(i);
  }
}

void main() {
  try {
    var result = squareRoot(-4);
    print("result: $result");
  } on NegativeSquareRootException catch (e) {
    print("Oops, Negative Number: $e");
  } catch (e) {
    print(e);
  } finally {
    print('Job Completed!');
  }
}
```

**Output:**
```
Oops, Negative Number: Sqauare root of negative number is not allowed here.
Job Completed!
```

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/exception_handeling-in-dart/
- **Fetched**: 2026-01-27
