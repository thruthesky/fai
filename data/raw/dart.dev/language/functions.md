# Functions in Dart - Complete Guide

## Overview

According to the documentation, "Dart is a true object-oriented language, so even functions are objects and have a type, Function." This fundamental characteristic means functions can be assigned to variables and passed as arguments to other functions.

## Basic Function Syntax

### Simple Function Declaration

```dart
bool isNoble(int atomicNumber) {
  return _nobleGases[atomicNumber] != null;
}
```

Functions work without type annotations, though the documentation recommends them for public APIs.

### Arrow Syntax

For single-expression functions, use the shorthand form:

```dart
bool isNoble(int atomicNumber) => _nobleGases[atomicNumber] != null;
```

The documentation clarifies that "only expressions can appear between the arrow and semicolon" — not statements.

## Parameters

Dart supports three parameter types: required positional, named, and optional positional parameters.

### Named Parameters

Named parameters are optional by default. Define them using curly braces:

```dart
void enableFlags({bool? bold, bool? hidden}) {
  ...
}
```

Call them with parameter names:

```dart
enableFlags(bold: true, hidden: false);
```

**Default values:**

```dart
void enableFlags({bool bold = false, bool hidden = false}) {
  ...
}
enableFlags(bold: true);
```

**Required named parameters:**

```dart
const Scrollbar({super.key, required Widget child});
```

### Optional Positional Parameters

Mark parameters as optional using square brackets:

```dart
String say(String from, String msg, [String? device]) {
  var result = '$from says $msg';
  if (device != null) {
    result = '$result with a $device';
  }
  return result;
}
```

With default values:

```dart
String say(String from, String msg, [String device = 'carrier pigeon']) {
  var result = '$from says $msg with a $device';
  return result;
}
```

## The main() Function

Every Dart application requires a top-level `main()` function as the entry point:

```dart
void main() {
  print('Hello, World!');
}
```

For command-line applications accepting arguments:

```dart
void main(List<String> arguments) {
  print(arguments);
  assert(arguments.length == 2);
  assert(int.parse(arguments[0]) == 1);
  assert(arguments[1] == 'test');
}
```

The documentation notes you can "use the args library to define and parse command-line arguments."

## Functions as First-Class Objects

Functions can be passed as parameters:

```dart
void printElement(int element) {
  print(element);
}

var list = [1, 2, 3];
list.forEach(printElement);
```

Functions can be assigned to variables:

```dart
var loudify = (msg) => '!!! ${msg.toUpperCase()} !!!';
assert(loudify('hello') == '!!! HELLO !!!');
```

## Function Types

Explicitly specify function types for clarity:

```dart
void greet(String name, {String greeting = 'Hello'}) =>
    print('$greeting $name!');

void Function(String, {String greeting}) g = greet;
g('Dash', greeting: 'Howdy');
```

The documentation states that "in Dart, functions are first-class objects, meaning they can be assigned to variables, passed as arguments, and returned from other functions."

## Anonymous Functions

Unnamed functions (lambdas or closures) follow the same structure as named functions:

```dart
const list = ['apples', 'bananas', 'oranges'];

var uppercaseList = list.map((item) {
  return item.toUpperCase();
}).toList();

for (var item in uppercaseList) {
  print('$item: ${item.length}');
}
```

Shortened with arrow syntax:

```dart
var uppercaseList = list.map((item) => item.toUpperCase()).toList();
uppercaseList.forEach((item) => print('$item: ${item.length}'));
```

## Lexical Scope

Dart uses lexical scoping. The documentation explains that you can "follow the curly braces outwards to see if a variable is in scope":

```dart
bool topLevel = true;

void main() {
  var insideMain = true;

  void myFunction() {
    var insideFunction = true;

    void nestedFunction() {
      var insideNestedFunction = true;

      assert(topLevel);
      assert(insideMain);
      assert(insideFunction);
      assert(insideNestedFunction);
    }
  }
}
```

## Lexical Closures

A closure is "a function object that can access variables in its lexical scope when the function sits outside that scope":

```dart
Function makeAdder(int addBy) {
  return (int i) => addBy + i;
}

void main() {
  var add2 = makeAdder(2);
  var add4 = makeAdder(4);

  assert(add2(3) == 5);
  assert(add4(3) == 7);
}
```

## Tear-Offs

Referring to functions without parentheses creates a tear-off (closure with same parameters):

**Recommended approach:**

```dart
var charCodes = [68, 97, 114, 116];
charCodes.forEach(print);  // Function tear-off
```

**Avoid unnecessary lambda wrapping:**

```dart
charCodes.forEach((code) {
  print(code);
});
```

## Testing Function Equality

```dart
void foo() {}

class A {
  static void bar() {}
  void baz() {}
}

void main() {
  Function x;

  x = foo;
  assert(foo == x);

  x = A.bar;
  assert(A.bar == x);

  var v = A();
  var w = A();
  var y = w;
  x = w.baz;

  assert(y.baz == x);  // Same instance
  assert(v.baz != w.baz);  // Different instances
}
```

## Return Values

All functions return values. Without explicit return, the function implicitly returns null:

```dart
foo() {}
assert(foo() == null);
```

Multiple return values using records:

```dart
(String, int) foo() {
  return ('something', 42);
}
```

## Getters and Setters

Custom getters and setters allow computed property access:

```dart
String _secret = 'Hello';

String get secret {
  print('Getter was used!');
  return _secret.toUpperCase();
}

set secret(String newMessage) {
  print('Setter was used!');
  if (newMessage.isNotEmpty) {
    _secret = newMessage;
    print('New secret: "$newMessage"');
  }
}

void main() {
  print('Current message: $secret');
  secret = 'Dart is fun';
  print('New message: $secret');
}
```

The documentation explains the purpose: "create a clear separation between the client and the provider."

## Generators

### Synchronous Generators

Return an `Iterable` using `sync*`:

```dart
Iterable<int> naturalsTo(int n) sync* {
  int k = 0;
  while (k < n) yield k++;
}
```

### Asynchronous Generators

Return a `Stream` using `async*`:

```dart
Stream<int> asynchronousNaturalsTo(int n) async* {
  int k = 0;
  while (k < n) yield k++;
}
```

### Recursive Generators

Improve performance using `yield*`:

```dart
Iterable<int> naturalsDownFrom(int n) sync* {
  if (n > 0) {
    yield n;
    yield* naturalsDownFrom(n - 1);
  }
}
```

## External Functions

External functions have implementations defined separately:

```dart
external void someFunc(int i);
```

The documentation states that "external introduces type information for foreign functions or values, making them usable in Dart" and that "implementation and usage is heavily platform specific." These are commonly used in interop contexts with C, JavaScript, or other languages.

External functions can be top-level functions, instance methods, getters, setters, non-redirecting constructors, or instance variables.

---

**Document Source:** Dart.dev Language Documentation
