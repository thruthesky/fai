# Callable Objects in Dart

## Overview

Dart allows instances of classes to be invoked like functions by implementing the `call()` method. This feature enables objects to emulate function behavior while maintaining the structure and capabilities of a class.

## How It Works

According to the documentation, "To allow an instance of your Dart class to be called like a function, implement the `call()` method."

The `call()` method supports the same features as regular functions, including:
- Parameters of various types
- Return type declarations
- Full function functionality

## Example

The documentation provides this practical example:

```dart
class WannabeFunction {
  String call(String a, String b, String c) => '$a $b $c!';
}

var wf = WannabeFunction();
var out = wf('Hi', 'there,', 'gang');

void main() => print(out);
```

This demonstrates a class that concatenates three string parameters with spaces between them and appends an exclamation mark. The instance `wf` can be called directly using function-call syntax: `wf('Hi', 'there,', 'gang')`.

## Key Benefit

This pattern enables objects to behave like functions, providing flexibility in object-oriented programming while maintaining the structure and state management advantages of classes.
