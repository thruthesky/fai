# Future In Dart

## Overview

"In dart, the Future represents a value or error that is not yet available." Futures enable you to work with asynchronous operations that will produce results at a later time.

## Creating Futures

### Using Future.delayed()
```dart
Future<String> getUserName() async {
  return Future.delayed(Duration(seconds: 2), () => 'Mark');
}
```

### Using Future.value()
```dart
Future<String> getUserName() {
  return Future.value('Mark');
}
```

## Using Futures with then()

The `then()` method allows you to handle a completed Future:

```dart
Future<String> getUserName() async {
  return Future.delayed(Duration(seconds: 2), () => 'Mark');
}

void main() {
  print("Start");
  getUserName().then((value) => print(value));
  print("End");
}
```

**Output:**
```
Start
End
Mark
```

## Future States

"Future represents the result of an asynchronous operation and can have 2 states."

### Uncompleted
When an async function is called, it returns an uncompleted Future that waits for the operation to finish or throw an error.

### Completed
A Future completes either with a value or with an error. The generic type indicates the result (`Future<int>`, `Future<String>`, `Future<void>`).

## Example with Async/Await

```dart
void main() {
  print("Start");
  getData();
  print("End");
}

void getData() async{
  String data = await middleFunction();
  print(data);
}

Future<String> middleFunction(){
  return Future.delayed(Duration(seconds:5), ()=> "Hello");
}
```

**Output:**
```
Start
End
Hello
```

**Note:** The sequence prints "Start," then "End," followed by "Hello" after 5 seconds.

---

## Source

- **URL**: https://dart-tutorial.com/asynchronous-programming/future-in-dart/
- **Fetched**: 2026-01-27
