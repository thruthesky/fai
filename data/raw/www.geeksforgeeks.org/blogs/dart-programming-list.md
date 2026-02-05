# Dart Programming - List Tutorial

## Overview

In Dart programming, List data type is similar to arrays in other programming languages. Lists represent ordered collections of objects, with the core libraries managing creation and manipulation.

## Logical Representation

Each list element has an index indicating its position. Elements are accessed and displayed by referencing their index value.

## Types of Lists by Length

### 1. Fixed Length List

A fixed-length list has a predetermined size that cannot change during runtime.

**Syntax:**
```dart
List ? list_Name = List.filled(number of elements, E, growable:boolean);
```

**Example:**
```dart
void main() {
    List? gfg = List.filled(5, null, growable: false);
    gfg[0] = 'Geeks';
    gfg[1] = 'For';
    gfg[2] = 'Geeks';

    print(gfg);
    print(gfg[2]);
}
```

**Output:**
```
[Geeks, For, Geeks, null, null]
Geeks
```

### 2. Growable List

Growable lists lack an initial size declaration and can expand during runtime.

**Adding a Single Value:**
```dart
void main() {
    var gfg = ['Geeks', 'For'];
    print(gfg);

    gfg.add('Geeks');
    print(gfg);
}
```

**Output:**
```
[Geeks, For]
[Geeks, For, Geeks]
```

**Adding Multiple Values:**
```dart
void main() {
    var gfg = ['Geeks'];
    print(gfg);

    gfg.addAll(['For', 'Geeks']);
    print(gfg);
}
```

**Output:**
```
[Geeks]
[Geeks, For, Geeks]
```

**Inserting at Specific Index:**
```dart
void main() {
    var gfg = ['Geeks', 'Geeks'];
    print(gfg);

    gfg.insert(1, 'For');
    print(gfg);
}
```

**Output:**
```
[Geeks, Geeks]
[Geeks, For, Geeks]
```

**Inserting Multiple Values at Index:**
```dart
void main() {
    var gfg = ['Geeks'];
    print(gfg);

    gfg.insertAll(1, ['For', 'Geeks']);
    print(gfg);
    print(gfg[1]);
}
```

**Output:**
```
[Geeks]
[Geeks, For, Geeks]
For
```

## Multidimensional Lists

### 2-Dimensional (2-D) List

2-D lists have table-like structure with rows and columns.

**Creation Method 1:**
```dart
void main() {
    int a = 3;
    int b = 3;

    var gfg = List.generate(a, (i) => List.filled(b, 0), growable: false);
    print(gfg);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            gfg[i][j] = i + j;
        }
    }
    print(gfg);
}
```

**Output:**
```
[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

**Creation Method 2:**
```dart
void main() {
    var gfg = List.generate(3, (i) => List.generate(3, (j) => i + j));
    print(gfg);
}
```

**Output:**
```
[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

### 3-Dimensional (3-D) List

```dart
void main() {
    var gfg = List.generate(3, (i) => List.generate(3,
        (j) => List.generate(3, (k) => i + j + k)));

    print(gfg);
}
```

**Output:**
```
[[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
 [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
 [[2, 3, 4], [3, 4, 5], [4, 5, 6]]]
```

## Key Takeaway

In a similar fashion one can create an n-dimensional List by using the List.generate() method. This approach scales to any dimensionality needed for complex data structures.

---

## Source

- **URL**: https://www.geeksforgeeks.org/blogs/dart-programming-list/
- **Fetched**: 2026-01-27
