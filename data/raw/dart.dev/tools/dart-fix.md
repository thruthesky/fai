# dart fix - Dart Command-Line Tool

## Overview

The `dart fix` command identifies and resolves two categories of issues:

1. **Analysis issues** detected by `dart analyze` that have automated fixes available
2. **Outdated API usages** when upgrading to newer Dart and Flutter SDK versions

## Applying Fixes

### Preview Changes
To see proposed modifications without making changes:

```bash
$ dart fix --dry-run
```

### Apply Changes
To implement the proposed modifications:

```bash
$ dart fix --apply
```

## Customizing Behavior

The `dart fix` command applies fixes only when diagnostics identify problems. While some diagnostics (like compilation errors) are automatically enabled, others such as lints require explicit configuration in the analysis options file.

Enabling additional lints can increase the number of available fixes, though not all diagnostics have associated fixes.

### Configuration Example

Consider this Dart class hierarchy:

```dart
class Vector2d {
  final double x, y;
  Vector2d(this.x, this.y);
}

class Vector3d extends Vector2d {
  final double z;
  Vector3d(final double x, final double y, this.z) : super(x, y);
}
```

Dart 2.17 introduced super initializers, allowing more concise constructor syntax:

```dart
class Vector3d extends Vector2d {
  final double z;
  Vector3d(super.x, super.y, this.z);
}
```

To enable `dart fix` for this upgrade:

**analysis_options.yaml:**
```yaml
linter:
  rules:
    - use_super_parameters
```

**pubspec.yaml:**
```yaml
environment:
  sdk: ">=2.17.0 <4.0.0"
```

Running the command then displays:
```
$ dart fix --dry-run
Computing fixes in myapp (dry run)... 9.0s

1 proposed fixes in 1 files.

lib/myapp.dart
  use_super_parameters • 1 fix
```

## VS Code Integration

The Dart plugin in VS Code automatically scans projects for repairable issues and displays notifications when found. After running `dart pub get` or `dart pub upgrade`, additional prompts may appear if package updates introduce new fixable issues.

**Important:** Save all files before running `dart fix` to ensure the tool uses current file versions.
