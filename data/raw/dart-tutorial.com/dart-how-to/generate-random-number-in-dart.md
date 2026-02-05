# Generate Random Numbers in Dart

## Overview

"This tutorial will teach you how to generate random numbers in dart programming. You will learn to generate random numbers between a range, and also generate a list of random numbers."

## Why Use Random Numbers

- **Random Number Games**: Create interactive games that rely on unpredictable values
- **Card Shuffling**: Randomize card deck arrangements for card-based applications

## Basic Example: Random Numbers 0-9 and 1-10

The first approach demonstrates generating numbers within specific ranges:

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

Key points:
- `random.nextInt(10)` generates values from 0 to 9
- Adding 1 to the result shifts the range to 1 to 10

## Universal Formula for Any Range

To generate random numbers between arbitrary minimum and maximum values:

```
min + Random().nextInt((max + 1) - min);
```

## Example: Random Numbers Between 10-20

This implementation applies the formula to a specific range:

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

---

## Source

- **URL**: https://dart-tutorial.com/dart-how-to/generate-random-number-in-dart/
- **Fetched**: 2026-01-27
