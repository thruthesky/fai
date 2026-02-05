# Null Safety in Dart

## Overview

"Null safety is a feature in the Dart programming language that helps developers to avoid null errors."
Also referred to as Sound Null Safety, this mechanism allows developers to identify potential null-related
issues during the editing phase rather than at runtime.

## Key Advantages

- Write safer, more reliable code
- Minimize application crashes caused by null reference errors
- Simplify debugging and bug-fixing processes

> Note: This feature prevents null errors, runtime bugs, vulnerabilities, and system crashes that are
> difficult to identify and resolve.

## Core Principles

### Non-Nullable by Default

In Dart, variables and fields cannot hold null values unless explicitly permitted. This represents a
fundamental shift in how developers must approach variable declarations.

```dart
int productid = 20; // non-nullable - valid
int productid = null; // error - not allowed
```

### Declaring Nullable Variables

To explicitly allow null values, append the `?` operator to the type:

```dart
String? name; // can be null or a String
```

## Usage Patterns

### Assigning Values to Nullable Variables

```dart
void main(){
  String? name;
  name = "John";    // valid assignment
  name = null;      // valid assignment to nullable type
}
```

### Working with Nullable Variables

Three primary techniques exist for handling nullable variables:

**1. Conditional Checking**
```dart
if(name == null){
  print("Name is null");
}
```

**2. Null Coalescing Operator (`??`)**
```dart
String name1 = name ?? "Stranger"; // assigns default if null
```

**3. Non-null Assertion Operator (`!`)**
```dart
String name2 = name!; // asserts value is not null
```

## Advanced Applications

### Lists with Nullable Elements

```dart
List<int?> items = [1, 2, null, 4]; // list containing nullable integers
print(items); // [1, 2, null, 4]
```

### Function Parameters

Functions can declare nullable parameters:

```dart
void printAddress(String? address) {
  print(address);
}

void main() {
  printAddress(null); // works - prints null
}
```

### Class Properties

Classes can define nullable properties:

```dart
class Person {
  String? name;
  Person(this.name);
}

void main() {
  Person person = Person(null); // valid
}
```

### Comprehensive Class Example

```dart
class Profile {
  String? name;
  String? bio;

  Profile(this.name, this.bio);

  void printProfile() {
    print("Name: ${name ?? "Unknown"}");
    print("Bio: ${bio ?? "None provided"}");
  }
}

void main() {
  Profile profile1 = Profile("John", "Software engineer");
  profile1.printProfile();
  // Output:
  // Name: John
  // Bio: Software engineer

  Profile profile2 = Profile(null, null);
  profile2.printProfile();
  // Output:
  // Name: Unknown
  // Bio: None provided
}
```

## Important Concepts

- Null represents the absence of a value
- "Common error in programming is caused due to null"
- Dart 2.12 introduced sound null safety to address null-related problems
- Non-nullable types are guaranteed never to be null
- NNBD (Non-Nullable By Default) prevents null assignment unless explicitly permitted

---

## Source

- **URL**: https://dart-tutorial.com/null-safety/null-safety-in-dart/
- **Fetched**: 2026-01-27
