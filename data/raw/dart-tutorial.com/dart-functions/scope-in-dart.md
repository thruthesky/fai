# Scope in Dart

## Scope In Dart

The scope concept refers to where values can be accessed or referenced. "Dart uses curly braces **{}** to determine the scope of variables. If you define a variable inside curly braces, you can't use it outside the curly braces."

## Method Scope

Variables created within a method are accessible only within that method's block, not outside it.

### Example 1: Method Scope

```dart
void main() {
  String text = "I am text inside main. Anyone can't access me.";
  print(text);
}
```

**Output:**
```
I am text inside main. Anyone can't access me.
```

In this example, the `text` variable can only be accessed and printed within the `main()` function.

## Global Scope

"You can define a variable in the global scope to use the variable anywhere in your program."

### Example 2: Global Scope

```dart
String global = "I am Global. Anyone can access me.";
void main() {
  print(global);
}
```

**Output:**
```
I am Global. Anyone can access me.
```

The `global` variable is a top-level variable accessible throughout the entire program.

## Lexical Scope

"Dart is lexically scoped language, which means you can find the scope of variables with the help of **braces {}**."

## Best Practice Note

"Define your variable as much as close **Local** as you can. It makes your code clean and prevents you from using or changing them where you shouldn't."

---

## Source

- **URL**: https://dart-tutorial.com/dart-functions/scope-in-dart/
- **Fetched**: 2026-01-27
