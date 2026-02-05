# dart:math Library Documentation

## Overview

The `dart:math` library provides essential mathematical operations. As stated in the documentation, it "provides common functionality such as sine and cosine, maximum and minimum, and constants such as pi and e."

To use this library, import it:

```dart
import 'dart:math';
```

## Trigonometry

The library includes basic trigonometric functions that operate using radians rather than degrees:

```dart
// Cosine
assert(cos(pi) == -1.0);

// Sine
var degrees = 30;
var radians = degrees * (pi / 180);
var sinOf30degrees = sin(radians);
// sin 30° = 0.5
assert((sinOf30degrees - 0.5).abs() < 0.01);
```

**Important:** These functions use radians, not degrees.

## Maximum and Minimum

Use `max()` and `min()` methods for comparing values:

```dart
assert(max(1, 1000) == 1000);
assert(min(1, -1000) == -1000);
```

## Math Constants

Access common mathematical constants:

```dart
print(e);      // 2.718281828459045
print(pi);     // 3.141592653589793
print(sqrt2);  // 1.4142135623730951
```

## Random Numbers

Generate random values using the `Random` class:

```dart
var random = Random();
random.nextDouble(); // Between 0.0 and 1.0: [0, 1)
random.nextInt(10);  // Between 0 and 9
random.nextBool();   // true or false
```

**Warning:** The default `Random` is unsuitable for cryptographic purposes. Use `Random.secure()` for cryptographically secure random generation.

## Additional Resources

Consult the API reference for complete method listings and related classes like `num`, `int`, and `double`.
