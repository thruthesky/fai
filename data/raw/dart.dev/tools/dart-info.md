# Dart Info Command-Line Tool

## Overview

The `dart info` command provides diagnostic information about your Dart tooling installation, active Dart processes, and project details when run in a directory containing a `pubspec.yaml` file.

## Basic Usage

Execute the command from any directory:

```bash
$ dart info
```

## Output Sections

### General Information

The output displays system and installation details, including:

- Dart version and channel (stable/dev)
- Operating system and version
- System locale

Example output on macOS:

```
#### General info

- Dart 2.19.2 (stable) (Tue Feb 7 18:37:17 2023 +0000) on "macos_arm64"
- on macos / Version 13.1 (Build 22C65)
- locale is en-US
```

### Process Information

Running processes appear in a formatted table showing memory usage, CPU consumption, elapsed time, and command-line arguments.

### Project Information

When executed in a project directory with `pubspec.yaml`, additional project details are included:

- SDK version constraints
- Dependency listings
- Development dependency listings

Example:

```
#### Project info

- sdk constraint: '>=2.19.2 <3.0.0'
- dependencies: path
- dev_dependencies: lints, test
```

## Advanced Options

Use the `--no-remove-file-paths` flag to preserve file paths and path dependencies in the output:

```bash
$ dart info --no-remove-file-paths
```

## Privacy Consideration

The documentation recommends reviewing output before including it in public bug reports to ensure no sensitive information is shared.
