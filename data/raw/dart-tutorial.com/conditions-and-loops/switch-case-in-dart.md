# Switch Case in Dart

## Switch Case In Dart

A switch case executes code blocks conditionally based on matching expression values. The basic structure evaluates an expression once and compares it against multiple case values.

### Syntax

```dart
switch(expression) {
    case value1:
        // statements
        break;
    case value2:
        // statements
        break;
    case value3:
        // statements
        break;
    default:
       // default statements
}
```

### How It Works

The mechanism operates as follows:

- The expression evaluates once, then compares against each case value
- When the expression matches a case value, those statements execute
- The `break` keyword exits the switch, preventing fall-through execution
- Unmatched expressions trigger the default block

**Note:** This construct serves as an alternative to chained if-else conditions.

## If Else If vs Switch

### Example: Using If Else If

```dart
void main(){
   var dayOfWeek = 5;
if (dayOfWeek == 1) {
        print("Day is Sunday.");
  }
else if (dayOfWeek == 2) {
       print("Day is Monday.");
     }
else if (dayOfWeek == 3) {
      print("Day is Tuesday.");
     }
else if (dayOfWeek == 4) {
        print("Day is Wednesday.");
     }
else if (dayOfWeek == 5) {
        print("Day is Thursday.");
   }
else if (dayOfWeek == 6) {
        print("Day is Friday.");
    }
else if (dayOfWeek == 7) {
        print("Day is Saturday.");
}else{
        print("Invalid Weekday.");
     }
}
```

**Output:** `Day is Thursday.`

### Example: Using Switch Statement

```dart
void main() {
  var dayOfWeek = 5;
  switch (dayOfWeek) {
    case 1:
        print("Day is Sunday.");
        break;
    case 2:
        print("Day is Monday.");
      break;
    case 3:
      print("Day is Tuesday.");
      break;
    case 4:
        print("Day is Wednesday.");
      break;
    case 5:
        print("Day is Thursday.");
      break;
    case 6:
        print("Day is Friday.");
      break;
    case 7:
        print("Day is Saturday.");
      break;
    default:
        print("Invalid Weekday.");
      break;
  }
}
```

**Output:** `Day is Thursday.`

**Note:** The switch syntax offers cleaner, more readable code compared to multiple if-else chains.

## Switch Case On Strings

Switch statements work with string values as well:

```dart
void main() {
 const weather = "cloudy";

  switch (weather) {
    case "sunny":
        print("Its a sunny day. Put sunscreen.");
        break;
    case "snowy":
        print("Get your skis.");
      break;
    case "cloudy":
    case "rainy":
      print("Please bring umbrella.");
      break;
    default:
        print("Sorry I am not familiar with such weather.");
      break;
  }
}
```

**Output:** `Please bring umbrella.`

## Switch Case On Enum

An enumeration (enum) defines custom types with finite options.

### Enum Syntax

```dart
enum enum_name {
  constant_value1,
  constant_value2,
  constant_value3
}
```

### Example: Switch Using Enum

```dart
// define enum outside main function
enum Weather{ sunny, snowy, cloudy, rainy}
// main method
void main() {
 const weather = Weather.cloudy;

  switch (weather) {
    case Weather.sunny:
        print("Its a sunny day. Put sunscreen.");
        break;
    case Weather.snowy:
        print("Get your skis.");
      break;
    case Weather.rainy:
    case Weather.cloudy:
      print("Please bring umbrella.");
      break;
    default:
        print("Sorry I am not familiar with such weather.");
      break;
  }
}
```

**Output:** `Please bring umbrella.`

Enums pair effectively with switch statements, offering type-safe options for conditional logic.

---

## Source

- **URL**: https://dart-tutorial.com/conditions-and-loops/switch-case-in-dart/
- **Fetched**: 2026-01-27
