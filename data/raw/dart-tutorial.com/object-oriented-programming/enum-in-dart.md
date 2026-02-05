# Enum in Dart

## Definition

"An enum is a special type that represents a fixed number of constant values. An enum is declared using the keyword **enum** followed by the enum's name."

## Syntax

```dart
enum enumName {
  constantName1,
  constantName2,
  constantName3,
  ...
  constantNameN
}
```

## Basic Example

Here's a practical demonstration using days of the week:

```dart
enum days {
  Sunday,
  Monday,
  Tuesday,
  Wednesday,
  Thrusday,
  Friday,
  Saturday
}

void main() {
  var today = days.Friday;
  switch (today) {
    case days.Sunday:
      print("Today is Sunday.");
      break;
    case days.Monday:
      print("Today is Monday.");
      break;
    case days.Tuesday:
      print("Today is Tuesday.");
      break;
    case days.Wednesday:
      print("Today is Wednesday.");
      break;
    case days.Thursday:
      print("Today is Thursday.");
      break;
    case days.Friday:
      print("Today is Friday.");
      break;
    case days.Saturday:
      print("Today is Saturday.");
      break;
  }
}
```

**Output:** `Today is Friday.`

## Using Enums in Classes

Enums work well as class properties:

```dart
enum Gender { Male, Female, Other }

class Person {
  String? firstName;
  String? lastName;
  Gender? gender;

  Person(this.firstName, this.lastName, this.gender);

  void display() {
    print("First Name: $firstName");
    print("Last Name: $lastName");
    print("Gender: $gender");
  }
}

void main() {
  Person p1 = Person("John", "Doe", Gender.Male);
  p1.display();

  Person p2 = Person("Menuka", "Sharma", Gender.Female);
  p2.display();
}
```

## Iterating Through All Values

Access all enum constants using the `.values` property:

```dart
enum Days { Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday }

void main() {
  for (Days day in Days.values) {
    print(day);
  }
}
```

## Enhanced Enums with Members

Enums can include data fields and constructors:

```dart
enum CompanyType {
  soleProprietorship("Sole Proprietorship"),
  partnership("Partnership"),
  corporation("Corporation"),
  limitedLiabilityCompany("Limited Liability Company");

  final String text;
  const CompanyType(this.text);
}

void main() {
  CompanyType soleProprietorship = CompanyType.soleProprietorship;
  print(soleProprietorship.text);
}
```

**Output:** `Sole Proprietorship`

## Key Advantages

- Define collections of named constants
- Enhance code clarity and maintainability
- Promote reusability across projects

## Core Characteristics

- Must contain at least one constant value
- Declared outside class definitions
- Suitable for storing numerous constant values

---

## Source

- **URL**: https://dart-tutorial.com/object-oriented-programming/enum-in-dart/
- **Fetched**: 2026-01-27
