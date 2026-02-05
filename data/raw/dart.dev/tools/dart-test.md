# dart test

## Overview

The `dart test` command executes tests that depend on the [`test` package](https://pub.dev/packages/test) located in a project's `test` directory. For Flutter projects, use `flutter test` instead.

## Basic Usage

Run all tests in the current project:

```bash
$ cd my_app
$ dart test
```

## Running Specific Tests

### By File Path

Specify particular test files or directories:

```bash
$ dart test test/library_tour/io_test.dart
00:00 +0: readAsString, readAsLines
00:00 +1: readAsBytes
...
```

### Using Name Filters

Use the `--name` (`-n`) flag to match test names:

```bash
$ dart test --name String
00:00 +0: test/library_tour/io_test.dart: readAsString, readAsLines
00:00 +1: test/library_tour/core_test.dart: print: print(nonString)
...
```

### Using Tags

Filter tests with `--tags` (`-t`) or exclude them with `--exclude-tags` (`-x`).

## Multiple Conditions

When combining multiple flags, only tests matching **all** conditions execute:

```bash
$ dart test --name String --name print
00:00 +0: test/library_tour/core_test.dart: print: print(nonString)
00:00 +1: test/library_tour/core_test.dart: print: print(String)
00:00 +2: All tests passed!
```

## Additional Options

The tool supports numerous flags controlling test execution, concurrency, timeouts, and output formatting. View all options using:

```bash
$ dart test --help
```

For comprehensive information, consult the [`test` package documentation](https://pub.dev/packages/test).
