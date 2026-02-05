# dart format

## Overview

The `dart format` command-line tool updates Dart source code to comply with established "Dart formatting guidelines." This tool delivers formatting consistent with what integrated development environments provide with Dart support.

## Specifying Files to Format

### Single Path
Provide one file or directory path. When given a directory, the formatter recurses through all subdirectories:

```bash
$ dart format .
```

### Multiple Paths
Use space-separated paths to format multiple locations:

```bash
$ dart format lib bin/updater.dart
```

### Preventing File Overwrites
By default, `dart format` modifies files in place. Control this behavior with flags:

- `--output` or `-o`: Prevents overwriting
- `-o show`: Displays formatted content without saving
- `-o json`: Returns formatted output as JSON
- `-o none`: Shows which files would change

```bash
$ dart format -o show bin/my_app.dart
```

## Exit Code Notifications

Adding `--set-exit-if-changed` makes the tool return exit code `1` when changes occur and `0` when none do. This enables CI/CD integration:

```bash
$ dart format -o none --set-exit-if-changed bin/my_app.dart
```

## Format on Save Configuration

### VS Code
Add this to `settings.json`:

```json
{
  "[dart]": {
    "editor.formatOnSave": true
  }
}
```

### IntelliJ and Android Studio
Follow JetBrains documentation for "Automatically reformat code on save" functionality.

## Formatting Behavior

The formatter performs these modifications:

- Removes excess whitespace
- Enforces 80-character line limits (or shorter)
- Adds trailing commas to multiline argument/parameter lists
- Removes trailing commas from single-line lists
- Repositions comments relative to commas

## Configurable Line Width

### Project-Level Configuration
Add to `analysis_options.yaml`:

```yaml
formatter:
  page_width: 123
```

### File-Level Override
Include this marker before any code:

```dart
// dart format width=123
```

*Note: Configurable page width requires language version 3.7 or later.*

## Additional Resources

For more command options:

```bash
$ dart help format
```

Reference the [dart_style package documentation](https://pub.dev/packages/dart_style) and the [formatter FAQ](https://github.com/dart-lang/dart_style/wiki/FAQ) for additional details on formatting decisions.
