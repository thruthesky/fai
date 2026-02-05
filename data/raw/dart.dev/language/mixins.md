# Mixins in Dart

## Overview

"Mixins are a way of defining code that can be reused in multiple class hierarchies." They enable you to add functionality to classes without relying on traditional inheritance.

## Basic Syntax

Use the `with` keyword to apply mixins to a class:

```dart
class Musician extends Performer with Musical {
  // ···
}

class Maestro extends Person with Musical, Aggressive, Demented {
  Maestro(String maestroName) {
    name = maestroName;
    canConduct = true;
  }
}
```

Define a mixin using the `mixin` keyword:

```dart
mixin Musical {
  bool canPlayPiano = false;
  bool canCompose = false;
  bool canConduct = false;

  void entertainMe() {
    if (canPlayPiano) {
      print('Playing piano');
    } else if (canConduct) {
      print('Waving hands');
    } else {
      print('Humming to self');
    }
  }
}
```

**Key constraints:** Mixins cannot have `extends` clauses and cannot declare generative constructors.

## Specifying Mixin Member Dependencies

### Abstract Methods

Declare abstract methods to ensure subclasses provide required implementations:

```dart
mixin Musician {
  void playInstrument(String instrumentName); // Abstract method.

  void playPiano() {
    playInstrument('Piano');
  }
  void playFlute() {
    playInstrument('Flute');
  }
}

class Virtuoso with Musician {
  @override
  void playInstrument(String instrumentName) {
    print('Plays the $instrumentName beautifully');
  }
}
```

### Accessing Subclass State

Use abstract getters to access properties in mixin subclasses:

```dart
mixin NameIdentity {
  String get name;

  @override
  int get hashCode => name.hashCode;

  @override
  bool operator ==(other) =>
      other is NameIdentity && name == other.name;
}

class Person with NameIdentity {
  final String name;
  Person(this.name);
}
```

### Interface Implementation

The `implements` clause ensures dependencies without actual implementation:

```dart
abstract interface class Tuner {
  void tuneInstrument();
}

mixin Guitarist implements Tuner {
  void playSong() {
    tuneInstrument();
    print('Strums guitar majestically.');
  }
}

class PunkRocker with Guitarist {
  @override
  void tuneInstrument() {
    print("Don't bother, being out of tune is punk rock.");
  }
}
```

### The `on` Clause

Use `on` to declare a superclass requirement when your mixin needs `super` calls:

```dart
class Musician {
  musicianMethod() {
    print('Playing music!');
  }
}

mixin MusicalPerformer on Musician {
  performerMethod() {
    print('Performing music!');
    super.musicianMethod();
  }
}

class SingerDancer extends Musician with MusicalPerformer { }
```

This ensures only subclasses of the specified type can use the mixin.

## Class vs. Mixin vs. Mixin Class

**`mixin`**: Defines code for reuse across hierarchies.

**`class`**: Declares a traditional class.

**`mixin class`**: Creates a single type usable as both a class and mixin (requires Dart 3.0+):

```dart
mixin class Musician {
  // ...
}

class Novice with Musician { } // Use as mixin
class Novice extends Musician { } // Use as class
```

Restrictions for mixin classes: cannot have `extends` or `with` clauses, and cannot have an `on` clause.
