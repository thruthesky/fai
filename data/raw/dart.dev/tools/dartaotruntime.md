# dartaotruntime

## Overview

The `dartaotruntime` command-line tool enables execution of ahead-of-time (AOT) compiled snapshots of Dart applications. This tool is compatible with Windows, macOS, and Linux platforms.

## Creating AOT Snapshots

To generate AOT snapshots, utilize the `aot-snapshot` subcommand from the `dart compile` command. This compilation approach produces pre-compiled Dart applications optimized for distribution and execution.

## Running AOT Applications

Execute AOT programs using the `dartaotruntime` command. Before usage, ensure that the path to your Dart installation's `bin` directory is included in your system's `PATH` environment variable.

## Practical Example

Here is a workflow demonstrating snapshot creation and execution:

**Step 1: Compile to AOT format**
```
$ dart compile aot-snapshot bin/myapp.dart
Generated: /Users/me/simpleapp/bin/myapp.aot
```

**Step 2: Run the compiled snapshot**
```
$ dartaotruntime bin/simpleapp.aot
```

## Accessing Help Information

To explore additional command-line options available with `dartaotruntime`, execute:

```
$ dartaotruntime --help
```

This displays comprehensive information about supported flags and configuration parameters for the runtime environment.

---

**Documentation Version:** Reflects Dart 3.10.3 (Last updated September 4, 2025)
