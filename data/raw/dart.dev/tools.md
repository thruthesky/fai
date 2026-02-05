# Dart Tools Documentation

## Overview

The Dart website provides comprehensive guidance on tools supporting the Dart language across all platforms. The page serves as a central resource for developers choosing appropriate development tools based on their application type.

## Getting Started by Application Type

| Application Type | Installation | Key Resources |
|---|---|---|
| Flutter (mobile, web, desktop) | [Install Flutter](https://docs.flutter.dev/get-started/install) | Flutter tools documentation |
| Web applications | Install Dart SDK | General-purpose and web-specific tools |
| Servers and CLI apps | Install Dart SDK | General-purpose and specialized tools |

**Important Note:** "The Flutter SDK includes the full Dart SDK."

## General-Purpose Tools

### DartPad

An browser-based editor requiring no installation, enabling users to learn Dart syntax and experiment with language features. It supports core libraries but excludes VM-specific libraries like `dart:io`.

### Integrated Development Environments

Officially supported:
- Android Studio
- IntelliJ IDEA and other JetBrains IDEs
- Visual Studio Code

Community-maintained options include Emacs, Vim, and Eclipse, plus Language Server Protocol support for compatible editors.

### Command-Line Interface

The `dart` tool provides unified CLI functionality for code creation, formatting, analysis, testing, documentation generation, compilation, execution, and package management through pub.

### Debugging Tools

Dart DevTools offers an integrated suite for debugging and performance analysis.

## Specialized Development Tools

**Web Development:** The `webdev` CLI builds and serves Dart web applications.

**Server/CLI Applications:**
- `dart run` executes uncompiled applications and snapshots
- `dartaotruntime` runs ahead-of-time compiled snapshots

## Page Information

- Current documentation version: Dart 3.10.3
- Last updated: July 7, 2025
- Source repository: GitHub dart-lang/site-www
