# dart doc - Markdown Content

## Overview

The `dart doc` command generates HTML reference documentation for Dart source code. This tool transforms documentation comments into comprehensive API reference websites.

## Writing Documentation

To include reference text and examples in generated documentation, use documentation comments with Markdown formatting. The official guidance recommends consulting the "[Effective Dart: Documentation](/effective-dart/documentation)" guide for best practices on composing doc comments.

## Generating API Documentation

### Prerequisites

Before generating documentation, ensure:
- You've run `dart pub get`
- Your package passes `dart analyze` without errors

### Basic Generation

Generate documentation by running `dart doc .` from your package root:

```bash
$ cd my_package
$ dart pub get
$ dart doc .
Documenting my_package...
...
Success! Docs generated into /Users/me/projects/my_package/doc/api
```

The output defaults to the `doc/api` directory. Modify this with the `--output` flag:

```bash
$ dart doc --output=api_docs .
```

### Testing Without Output

Test your package for documentation issues without saving generated files:

```bash
$ dart doc --dry-run .
```

### Configuration

Create a `dartdoc_options.yaml` file in your package root to customize generation behavior. For detailed configuration options, visit the official options documentation.

## Viewing Generated Documentation

### Local Viewing

Generated docs require an HTTP server. Activate and use `package:dhttpd`:

```bash
$ dart pub global activate dhttpd
$ dart pub global run dhttpd --path doc/api
```

Access the docs at `http://localhost:8080` (or the URL displayed).

### Online Hosting

Deploy generated docs using Firebase Hosting, GitHub Pages, or similar static hosting services.

### pub.dev Documentation

The package repository automatically generates and hosts documentation for uploaded packages. Access API reference via each package's info box on pub.dev.

### Dart SDK Documentation

Core library documentation is available at:
- **Stable**: api.dart.dev
- **Beta**: api.dart.dev/beta
- **Dev**: api.dart.dev/dev
- **Main**: api.dart.dev/main

## Troubleshooting

### Search Bar Issues

If search functionality fails, verify:
1. Docs are served via HTTP server, not accessed directly
2. The `index.json` file exists and is accessible

### Sidebar Loading Problems

Common causes include:
1. Docs aren't being served via HTTP server
2. Deprecated base-href configuration is present (remove it)

### Missing API Documentation

Verify:
1. The API is exposed as public (check package layout for public libraries)
2. URLs use correct case-sensitivity matching source declarations

### Icon Display Issues

Material Symbols font may not be loading. Solutions include:
1. Using a proxy for Google Fonts access
2. Updating pages to use local font files

---

**Last Updated**: 2024-04-11 | **Dart Version**: 3.10.3
