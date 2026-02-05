# For Loop in Dart

## For Loop

This represents the most frequently used looping construct. A **for loop** executes a code block repeatedly based on specified conditions. The structure follows this pattern:

```dart
for(initialization; condition; increment/decrement){
    statements;
}
```

Key components:
- **Initialization** executes once before the loop begins
- **Condition** determines whether the block executes
- **Increment/Decrement** runs after each iteration

## Example 1: Print 1 To 10

Demonstrates basic iteration from 1 to 10 with `int i = 1`, condition `i<=10`, and increment `i++`:

```dart
void main() {
  for (int i = 1; i <= 10; i++) {
    print(i);
  }
}
```

**Output:** Numbers 1 through 10, each on separate lines

## Example 2: Print 10 To 1

Reverses the sequence using initialization `int i = 10`, condition `i>=1`, and decrement `i--`:

```dart
void main() {
  for (int i = 10; i >= 1; i--) {
    print(i);
  }
}
```

**Output:** Numbers 10 down to 1

## Example 3: Repeat Text 10 Times

Prints a name repeatedly based on loop iterations:

```dart
void main() {
  for (int i = 0; i < 10; i++) {
    print("John Doe");
  }
}
```

**Output:** "John Doe" appears 10 times

## Example 4: Sum Natural Numbers

Calculates the total of n natural numbers using accumulation:

```dart
void main(){
  int total = 0;
  int n = 100;

  for(int i=1; i<=n; i++){
    total = total + i;
  }

  print("Total is $total");
}
```

**Output:** `Total is 5050`

## Example 5: Display Even Numbers

Filters and displays even numbers within a specific range:

```dart
void main(){
  for(int i=50; i<=100; i++){
    if(i%2 == 0){
      print(i);
    }
  }
}
```

**Output:** Even numbers from 50 to 100 (50, 52, 54... through 100)

## Infinite Loop In Dart

An infinite loop occurs when the condition never evaluates to false, continuously consuming system resources. Example:

```dart
void main() {
  for (int i = 1; i >= 1; i++) {
    print(i);
  }
}
```

**Warning:** "Infinite loops consume computer resources continuously, use more power, and slow your system. Always verify your loop logic before execution."

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/for-loop-in-dart/
- **Fetched**: 2026-01-27
