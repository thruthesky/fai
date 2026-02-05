# How to Use Packages

## Overview

The Dart ecosystem manages shared software through packages via the **pub package manager**. Publicly available packages are hosted on [pub.dev](https://pub.dev), though packages can also be loaded from local file systems or Git repositories. Pub handles version dependencies, ensuring compatible package versions across your project and SDK.

## Key Steps to Using Packages

### 1. Creating a pubspec

A package requires a `pubspec.yaml` file in its root directory. The simplest version contains only a package name:

```yaml
name: my_app
```

To declare dependencies on hosted packages:

```yaml
name: my_app

dependencies:
  intl: ^0.20.2
  path: ^1.9.1
```

You can also add dependencies programmatically:

```bash
$ dart pub add vector_math
```

### 2. Getting Packages

Run `dart pub get` from your application's root directory:

```bash
$ cd <path-to-my_app>
$ dart pub get
```

This process retrieves dependencies from the system cache or downloads them from pub.dev. Pub creates a `package_config.json` file that maps package names to their cached locations.

### 3. Importing Libraries

Use the `package:` prefix to import libraries:

```dart
import 'package:js/js.dart' as js;
import 'package:intl/intl.dart';
```

Within your own packages, use the same format:

```dart
import 'package:transmogrify/parser.dart';
```

### 4. Upgrading Dependencies

Pub locks dependency versions in a `pubspec.lock` file. To update to newer versions:

```bash
$ dart pub upgrade
```

To upgrade a single package:

```bash
$ dart pub upgrade transmogrify
```

## Production Deployment

For production environments, enforce the lockfile:

```bash
$ dart pub get --enforce-lockfile
```

This ensures deployment uses only tested versions matching your `pubspec.lock` file exactly, preventing unexpected dependency changes.

## Additional Resources

- **Creating packages**: [Guide](/tools/pub/create-packages)
- **Publishing packages**: [Guide](/tools/pub/publishing)
- **Dependencies reference**: [Pub dependencies](/tools/pub/dependencies)
- **Pubspec format**: [Documentation](/tools/pub/pubspec)
- **Troubleshooting**: [Pub troubleshooting guide](/tools/pub/troubleshoot)
