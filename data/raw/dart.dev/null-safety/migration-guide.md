# Migrating to Null Safety in Dart

## Overview

This guide explains how to migrate existing Dart code to null safety. The process involves five key steps: waiting for dependencies, migrating code, analyzing, testing, and publishing.

## Step 1: Wait to Migrate

Before beginning migration, ensure your package's dependencies support null safety. The Dart team recommends migrating code in dependency order—starting with "leaf" packages that have no intra-package dependencies.

### Switch to Dart 2.19.6

The migration tool is only available in Dart 2.19.6 (the final release supporting null-safety migration). Verify your version:

```bash
$ dart --version
Dart SDK version: 2.19.6
```

### Check Dependency Status

Use this command to identify which dependencies support null safety:

```bash
$ dart pub outdated --mode=null-safety
```

This displays packages and their null-safety support status. All direct and dev dependencies must support null safety before proceeding.

### Update Dependencies

Upgrade to null-safe versions:

```bash
$ dart pub upgrade --null-safety
$ dart pub get
```

## Step 2: Migrate

Two approaches exist for migration: using the automated tool or migrating manually.

### Using the Migration Tool

Start the interactive migration tool:

```bash
$ dart migrate
```

The tool opens a web interface displaying proposed changes. Key features include:

- **Viewing suggestions**: See inferred nullability for each variable and type
- **Understanding results**: Click line numbers to see reasoning behind changes
- **Improving results**: Use hint markers to guide the tool's inference

#### Hint Markers Reference

| Marker | Effect |
|--------|--------|
| `/*! */` | Casts expression to non-nullable type |
| `/*?*/` | Marks preceding type as nullable |
| `/*late*/` | Marks variable with late initialization |
| `/*required*/` | Marks parameter as required |

#### Example with Hint Markers

Without hints:
```dart
var zero = ints[0];
var one = zero! + 1;
```

With `/*!*/` hint on the assignment:
```dart
var zero = ints[0]/*!*/;
var one = zero + 1;
```

#### Opting Out Files

For large projects, you can exclude specific files from migration by clicking their checkboxes. These files receive a version comment but remain unchanged. However, only fully migrated packages are compatible with Dart 3.

#### Applying Changes

Once satisfied with proposed changes, click "Apply migration." The tool removes hint markers and updates the SDK constraint in pubspec.yaml.

### Migrating by Hand

For those preferring manual migration:

1. **Update pubspec.yaml** with minimum SDK constraint:
   ```yaml
   environment:
     sdk: '>=2.12.0 <3.0.0'
   ```

2. **Regenerate package configuration**:
   ```bash
   $ dart pub get
   ```

3. **Open package in IDE** and address analysis errors by adding `?`, `!`, `required`, and `late` as needed.

4. **Migrate leaf libraries first**, then libraries depending on them.

## Step 3: Analyze

Perform static analysis to verify code correctness:

```bash
$ dart pub get
$ dart analyze
```

## Step 4: Test

Run your test suite to ensure migrated code functions properly:

```bash
$ dart test
```

Update tests expecting null values as necessary.

## Step 5: Publish

### Update Package Version

Indicate a breaking change by updating the version:
- For versions ≥1.0.0: increase major version (e.g., 2.3.2 → 3.0.0)
- For versions <1.0.0: increase minor version or reach 1.0.0

### Check Pubspec Requirements

Before publishing:
- Set Dart lower SDK constraint to at least 2.12.0
- Use stable versions of all direct dependencies

### Update Examples and Documentation

Ensure all examples, tutorials, and documentation reflect the null-safe release.

## Key Takeaway

Upon successful migration and with all dependencies migrated, you'll see this confirmation message:

```
Compiling with sound null safety
```

This indicates your program is protected against null-reference errors.
