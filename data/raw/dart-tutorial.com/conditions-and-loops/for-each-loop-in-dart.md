# For Each Loop in Dart

## Overview

"The **for each** loop iterates over all list elements or variables. It is useful when you want to loop through **list/collection**."

The basic syntax is:
```dart
collection.forEach(void f(value));
```

## Example 1: Print Each Item Of List Using Foreach

This demonstrates iterating through a list of football player names:

```dart
void main(){
  List<String> footballplayers=['Ronaldo','Messi','Neymar','Hazard'];
  footballplayers.forEach( (names)=>print(names));
}
```

**Output:**
```
Ronaldo
Messi
Neymar
Hazard
```

## Example 2: Print Each Total and Average Of Lists

This program calculates the sum and average of numeric values:

```dart
void main(){
  List<int> numbers = [1,2,3,4,5];

  int total = 0;

   numbers.forEach( (num)=>total= total+ num);

  print("Total is $total.");

  double avg = total / (numbers.length);

  print("Average is $avg.");
}
```

**Output:**
```
Total is 15.
Average is 3.
```

## For In Loop In Dart

An alternative looping mechanism that simplifies list iteration:

```dart
void main(){
    List<String> footballplayers=['Ronaldo','Messi','Neymar','Hazard'];

  for(String player in footballplayers){
    print(player);
  }
}
```

**Output:**
```
Ronaldo
Messi
Neymar
Hazard
```

## How to Find Index Value Of List

"In dart, asMap method converts the list to a map where the keys are the index and values are the element at the index."

```dart
void main(){

List<String> footballplayers=['Ronaldo','Messi','Neymar','Hazard'];

footballplayers.asMap().forEach((index, value) => print("$value index is $index"));

}
```

**Output:**
```
Ronaldo index is 0
Messi index is 1
Neymar index is 2
Hazard index is 3
```

## Example 3: Print Unicode Value of Each Character of String

This demonstrates extracting Unicode values from characters:

```dart
void main(){

String name = "John";

for(var codePoint in name.runes){
  print("Unicode of ${String.fromCharCode(codePoint)} is $codePoint.");
}
}
```

**Output:**
```
Unicode of J is 74.
Unicode of o is 111.
Unicode of h is 104.
Unicode of n is 110.
```

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/for-each-loop-in-dart/
- **Fetched**: 2026-01-27
