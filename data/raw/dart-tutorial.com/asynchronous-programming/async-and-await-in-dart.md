# Async and Await In Dart

## Overview

"**Async/await** is a feature in Dart that allows us to write asynchronous code that looks and behaves like synchronous code, making it easier to read."

When you mark a function with `async`, it returns a Future object containing the work's result. The `await` keyword pauses execution until the awaited Future completes, enabling synchronous-looking asynchronous code.

## Key Concepts

- Add `async` before a function body to make it asynchronous
- The `await` keyword only works within async functions
- "You cannot perform an asynchronous operation from a synchronous function"

## Code Examples

### Example 1: Synchronous Function (Without async/await)

```dart
void main() {
  print("Start");
  getData();
  print("End");
}

void getData() {
  String data = middleFunction();
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
Instance of '_Future<String>'
```

### Example 2: Asynchronous Function (With async/await)

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

### Example 3: Error Handling

Use `try-catch` blocks to handle errors in async functions:

```dart
main() {
  print("Start");
  getData();
  print("End");
}

void getData() async{
    try{
        String data = await middleFunction();
        print(data);
    }catch(err){
        print("Some error $err");
    }
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

## Important Terms

- **async**: Keyword marking a function as asynchronous
- **async function**: Functions marked with the async keyword
- **await**: Retrieves completed output from an asynchronous expression

---

## Source

- **URL**: https://dart-tutorial.com/asynchronous-programming/async-and-await-in-dart/
- **Fetched**: 2026-01-27
