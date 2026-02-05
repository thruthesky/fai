# Math in Dart

## Overview

"Math helps you to perform mathematical calculations efficiently. With dart math, you can **generate random number**, **find square root**, **find power of number**, or **round specific numbers**." To utilize these features, you must `import 'dart:math';`.

## Generating Random Numbers

### Basic Random Number Generation

The following demonstrates creating random values from 0-9 and 1-10:

```dart
import 'dart:math';
void main()
{
Random random = new Random();
int randomNumber = random.nextInt(10); // from 0 to 9 included
print("Generated Random Number Between 0 to 9: $randomNumber");

int randomNumber2 = random.nextInt(10)+1; // from 1 to 10 included
print("Generated Random Number Between 1 to 10: $randomNumber2");
}
```

The `random.nextInt(10)` method generates integers from 0-9, while adding 1 produces values from 1-10.

### Random Numbers Within Specific Ranges

Use this formula for arbitrary ranges:

```
min + Random().nextInt((max + 1) - min);
```

### Example: Range 10-20

```dart
import 'dart:math';
void main()
{
int min = 10;
int max = 20;

int randomnum = min + Random().nextInt((max + 1) - min);

print("Generated Random number between $min and $max is: $randomnum");
}
```

**Output:** `Generated Random number between 10 and 20 is 19`

## Boolean and Double Values

Generate boolean and decimal values using:

```dart
Random().nextBool(); // return true or false
Random().nextDouble(); // return 0.0 to 1.0
```

### Example: Random Boolean and Double

```dart
import 'dart:math';
void main()
{
double randomDouble = Random().nextDouble();
bool randomBool = Random().nextBool();

print("Generated Random double value is: $randomDouble");
print("Generated Random bool value is: $randomBool");
}
```

### Example: List of Random Numbers

Generate a list of 10 random integers between 1-100:

```dart
import 'dart:math';
void main()
{
List<int> randomList = List.generate(10, (_) => Random().nextInt(100)+1);
print(randomList);
}
```

## Useful Math Functions

| Function | Output | Description |
|----------|--------|-------------|
| pow(10,2) | 100 | 10 raised to power 2 equals 100 |
| max(10,2) | 10 | Largest value between two numbers |
| min(10,2) | 2 | Smallest value between two numbers |
| sqrt(25) | 5 | Square root calculation |

### Comprehensive Example

```dart
import 'dart:math';
void main()
{
  int num1 = 10;
  int num2 = 2;

  num powernum = pow(num1,num2);
  num maxnum = max(num1,num2);
  num minnum = min(num1,num2);
  num squareroot = sqrt(25); // Square root of 25

  print("Power is $powernum");
  print("Maximum is $maxnum");
  print("Minimum is $minnum");
  print("Square root is $squareroot");

}
```

**Output:**
```
Power is 100
Maximum is 10
Minimum is 2
Square root is 5.0
```

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/math-in-dart/
- **Fetched**: 2026-01-27
