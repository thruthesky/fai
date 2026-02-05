# dart run

## Overview

The `dart run` command enables execution of Dart programs directly from the command line. It supports running files, packages, and dependencies without requiring separate tools.

**Basic syntax:**
```
dart run [options] [<DART_FILE> | <PACKAGE_TARGET>] [args]
```

**Quick example:**
```bash
$ dart create myapp
$ cd myapp
$ dart run
```

## Running a Dart File

Execute any Dart file using its relative path:

```bash
$ dart run tool/debug.dart
```

## Running Programs in Packages

### From Dependent Packages

Access programs in the `bin` directory of packages your project depends on. Omit the program name if it matches the package name:

```bash
$ dart run bar
```

For different program names, use the format `<package_name>:<program_name>`:

```bash
$ dart run bar:baz
```

### From Current Package

When in your package's top directory, omit the package name to run the main program:

```bash
$ dart run
```

For non-matching program names, prefix with a colon:

```bash
$ dart run :baz
```

For programs outside `bin`, provide the relative path:

```bash
$ dart run tool/debug.dart
```

## Passing Arguments to main()

Add arguments after the command:

```bash
$ dart run tool/debug.dart arg1 arg2
```

When running the current package's main program:

```bash
$ dart run foo arg1 arg2
```

## Debugging Options

**Enable assert statements:**
```bash
$ dart run --enable-asserts tool/debug.dart
```

**Enable DevTools integration:**
```bash
$ dart run --observe tool/debug.dart
```

Run `dart run --help` for additional debugging options.

## Experimental Features

Use [experiment flags](/tools/experiment-flags) to enable development features during testing.
