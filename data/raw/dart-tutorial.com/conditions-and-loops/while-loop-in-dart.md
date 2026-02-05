# While Loop in Dart

## While Loop

In a while loop, the body executes repeatedly as long as the condition remains true. The condition is evaluated before each iteration. If true, the code within the braces executes; if false, the loop terminates.

## Syntax

```dart
while(condition){
    //statement(s);
    // Increment (++) or Decrement (--) Operation;
}
```

**Key points:**
- The condition inside parentheses is evaluated first
- If true, code inside braces executes
- The condition is rechecked after each iteration
- When false, the loop stops

## Example 1: Print Numbers 1 to 10

This program demonstrates printing sequential numbers using a while loop.

```dart
void main() {
  int i = 1;
  while (i <= 10) {
    print(i);
    i++;
  }
}
```

**Output:**
```
1 2 3 4 5 6 7 8 9 10
```

**Important:** Always increment or decrement the loop variable. Failing to do so creates an infinite loop.

## Example 2: Print Numbers 10 to 1

This program counts downward using a while loop.

```dart
void main() {
  int i = 10;
  while (i >= 1) {
    print(i);
    i--;
  }
}
```

**Output:**
```
10 9 8 7 6 5 4 3 2 1
```

## Example 3: Sum of Natural Numbers

This program calculates the sum of the first 100 natural numbers (1+2+3+...+100).

```dart
void main(){
  int total = 0;
  int n = 100;
  int i = 1;

  while(i<=n){
    total = total + i;
    i++;
  }

  print("Total is $total");
}
```

**Output:**
```
Total is 5050
```

## Example 4: Display Even Numbers (50-100)

This program prints all even numbers within a specified range.

```dart
void main(){
  int i = 50;
  while(i<=100){
    if(i%2 == 0){
      print(i);
    }
    i++;
  }
}
```

**Output:**
```
50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100
```

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/while-loop-in-dart/
- **Fetched**: 2026-01-27
