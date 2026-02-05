# Dart Pub Command-Line Interface Documentation

## Overview

The `dart pub` command provides package management functionality for Dart projects. You can invoke it through either the `dart` or `flutter` tools:

```bash
$ dart pub get    # Gets dependencies for a non-Flutter package
$ flutter pub get # Gets dependencies for a Flutter package
```

**Note:** The standalone `pub` command has been discontinued; use `dart pub` or `flutter pub` instead.

## Subcommands List

Pub organizes its functionality into several key subcommands:

- `add` — Adds a package dependency
- `cache` — Manages the local package cache
- `deps` — Displays all project dependencies
- `downgrade` — Retrieves minimum dependency versions
- `get` — Fetches dependencies listed in pubspec
- `global` — Manages globally-available packages
- `outdated` — Identifies out-of-date dependencies
- `publish` — Uploads packages to pub.dev
- `remove` — Removes a package dependency
- `token` — Manages authentication tokens
- `unpack` — Extracts package contents
- `upgrade` — Updates dependencies to latest versions

## Subcommand Categories

### Managing Dependencies

The dependency management tools help maintain your project's package requirements:

**`cache`** — "Manages pub's local package cache. Use this subcommand to add packages to your cache, or to perform a clean reinstall of all packages in your cache."

**`get`** — Retrieves packages and creates or updates the lock file based on current constraints.

**`upgrade`** — "Retrieves the latest version of each package listed as dependencies used by the current package."

**`downgrade`** — Tests lower bounds by retrieving minimum dependency versions.

**`outdated`** — Analyzes which dependencies need updating and recommends actions.

**`deps`** — Lists all active project dependencies.

### Running Command-Line Apps

The `global` subcommand enables system-wide access to package executables after adding the cache `bin` directory to your system path.

### Deploying Packages

**Publishing** — Use `publish` to upload packages to pub.dev. Manage access through uploaders settings.

**Command-line Apps** — Include the `executables` tag in your pubspec to enable users to activate scripts globally via `dart pub global activate`.

## Global Options

These flags work with all subcommands:

| Option | Purpose |
|--------|---------|
| `--help` or `-h` | Display usage information |
| `--trace` | Show debugging output during errors |
| `--verbose` or `-v` | Equivalent to `--verbosity=all` |
| `--directory=<dir>` or `-C <dir>` | Execute command in specified directory |
| `--[no-]color` | Toggle colored output (default: enabled at terminals) |

## Version Information

Documentation reflects **Dart 3.10.3**. The `dart pub` command debuted in Dart 2.10.
