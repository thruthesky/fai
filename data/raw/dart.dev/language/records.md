# Records in Dart

Records are an anonymous, immutable, aggregate type that bundle multiple objects into a single object. Unlike other collection types, they feature fixed size, heterogeneous composition, and strong typing.

## Record Syntax

Record expressions use comma-delimited lists enclosed in parentheses:

```dart
var record = ('first', a: 2, b: true, 'last');
```

Record type annotations define return and parameter types similarly:

```dart
(int, int) swap((int, int) record) {
  var (a, b) = record;
  return (b, a);
}
```

Positional fields appear directly inside parentheses, while named fields use curly braces with type-name pairs:

```dart
// Positional fields
(String, int) record = ('A string', 123);

// Named fields
({int a, bool b}) record = (a: 123, b: true);
```

Named fields form part of the record's type definition. Two records with differently-named fields possess different types.

## Record Fields

Record fields are accessible through built-in getters. They remain immutable without setters.

Named fields expose same-name getters. Positional fields use `$<position>` notation, excluding named fields:

```dart
var record = ('first', a: 2, b: true, 'last');

print(record.$1); // 'first'
print(record.a);  // 2
print(record.b);  // true
print(record.$2); // 'last'
```

## Record Types

Records use structural typing based on field composition. The record's shape (field set, types, and names) determines its type uniquely.

Two records with identical shapes from unrelated sources share the same type, despite lacking coupling.

## Record Equality

Records equal when they share identical shape and field values. Named field order doesn't affect equality:

```dart
(int x, int y, int z) point = (1, 2, 3);
(int r, int g, int b) color = (1, 2, 3);

print(point == color); // true
```

However, named fields with different names produce unequal records despite matching values.

## Multiple Returns

Functions can return bundled values using records with pattern destructuring:

```dart
(String name, int age) userInfo(Map<String, dynamic> json) {
  return (json['name'] as String, json['age'] as int);
}

var (name, age) = userInfo(json);
```

## Records as Data Structures

Records serve as lightweight containers without requiring class declarations. They work well for homogeneous collections:

```dart
final buttons = [
  (
    label: "Button I",
    icon: const Icon(Icons.upload_file),
    onPressed: () => print("Action -> Button I"),
  ),
];
```

### Records and Typedefs

Type aliases improve readability and enable future implementation changes:

```dart
typedef ButtonItem = ({String label, Icon icon, void Function()? onPressed});
final List<ButtonItem> buttons = [...];
```

This approach allows transitioning from records to classes or extension types without updating consumer code.
