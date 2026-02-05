# Function Parameter in Dart

## Parameter In Dart

Parameters represent the mechanism for transferring values into functions. The quantity of arguments supplied must correspond to the number of parameters declared. Functions may accept any number of parameters.

```dart
// here a and b are parameters
void add(int a, int b) {
}
```

## Positional Parameter In Dart

With positional parameters, arguments must be supplied in the identical sequence as specified in the function definition. Reversing the order produces incorrect output.

### Example 1: Use Of Positional Parameter

The `printInfo` function requires two arguments in a specific order. Swapping them yields unintended results:

```dart
void printInfo(String name, String gender) {
  print("Hello $name your gender is $gender.");
}

void main() {
  // passing values in wrong order
  printInfo("Male", "John");

  // passing values in correct order
  printInfo("John", "Male");
}
```

**Output:**
```
Hello Male your gender is John.
Hello John your gender is Male.
```

### Example 2: Providing Default Value On Positional Parameter

This function accepts two required positional parameters and one optional parameter with a default assignment:

```dart
void printInfo(String name, String gender, [String title = "sir/ma'am"]) {
  print("Hello $title $name your gender is $gender.");
}

void main() {
  printInfo("John", "Male");
  printInfo("John", "Male", "Mr.");
  printInfo("Kavya", "Female", "Ms.");
}
```

**Output:**
```
Hello sir/ma'am John your gender is Male.
Hello Mr. John your gender is Male.
Hello Ms. Kavya your gender is Female.
```

### Example 3: Providing Default Value On Positional Parameter

The `add` function demonstrates optional parameters with initialization values:

```dart
void add(int num1, int num2, [int num3=0]){
   int sum;
  sum = num1 + num2 + num3;

   print("The sum is $sum");
}

void main(){
  add(10, 20);
  add(10, 20, 30);
}
```

**Output:**
```
The sum is 30
The sum is 60
```

## Named Parameter In Dart

Named parameters enhance code clarity by explicitly identifying each parameter's purpose. **Curly braces {}** denote named parameters, allowing arguments in any sequence.

### Example 1: Use Of Named Parameter

Arguments may be supplied in arbitrary order when using named parameters:

```dart
void printInfo({String? name, String? gender}) {
  print("Hello $name your gender is $gender.");
}

void main() {
  // you can pass values in any order in named parameters.
  printInfo(gender: "Male", name: "John");
  printInfo(name: "Sita", gender: "Female");
  printInfo(name: "Reecha", gender: "Female");
  printInfo(name: "Reecha", gender: "Female");
  printInfo(name: "Harry", gender: "Male");
  printInfo(gender: "Male", name: "Santa");
}
```

**Output:**
```
Hello John your gender is Male.
Hello Sita your gender is Female.
Hello Reecha your gender is Female.
Hello Reecha your gender is Female.
Hello Harry your gender is Male.
Hello Santa your gender is Male.
```

### Example 2: Use Of Required In Named Parameter

The `required` keyword enforces that specified parameters must be provided during function invocation:

```dart
void printInfo({required String name, required String gender}) {
  print("Hello $name your gender is $gender.");
}

void main() {
  // you can pass values in any order in named parameters.
  printInfo(gender: "Male", name: "John");
  printInfo(gender: "Female", name: "Suju");
}
```

**Output:**
```
Hello John your gender is Male.
Hello Suju your gender is Female.
```

## Optional Parameter In Dart

Optional parameters make arguments dispensable during function calls. **Square brackets []** designate optional parameters.

### Example: Use Of Optional Parameter

The `printInfo` function accepts two obligatory positional parameters and one discretionary parameter:

```dart
void printInfo(String name, String gender, [String? title]) {
  print("Hello $title $name your gender is $gender.");
}

void main() {
  printInfo("John", "Male");
  printInfo("John", "Male", "Mr.");
  printInfo("Kavya", "Female", "Ms.");
}
```

**Output:**
```
Hello  John your gender is Male.
Hello Mr. John your gender is Male.
Hello Ms. Kavya your gender is Female.
```

---

**Key Takeaway:** Named parameters accept arguments in any order, while the `?` operator addresses null-safety considerations discussed in subsequent chapters.

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/function-parameter-in-dart/
- **Fetched**: 2026-01-27
