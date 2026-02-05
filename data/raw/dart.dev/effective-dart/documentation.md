# Effective Dart: Documentation

## Overview

Documentation helps developers understand code. Clear, helpful comments save time for readers and future maintainers.

## Comments

### Format Comments Like Sentences

Capitalize the first word unless it's a case-sensitive identifier. End with a period.

### Avoid Block Comments for Documentation

Use // for all comments except temporarily commented-out code sections.

## Doc Comments

Doc comments use /// syntax and are parsed by dart doc to generate documentation.

### Use /// for Documentation

```dart
/// The number of characters in this chunk when unsplit.
int get length => ...
```

### Document Public APIs

Document most public libraries, top-level variables, types, and members.

### Start with Single-Sentence Summary

Begin with a brief, user-centric description ending in a period.

```dart
/// Deletes the file at [path] from the file system.
void delete(String path) { ... }
```

### Include Code Samples

```dart
/// The lesser of two numbers.
///
/// ```dart
/// min(5, 3) == 3
/// ```
num min(num a, num b) => ...
```

### Use Square Brackets for In-Scope Identifiers

```dart
/// Throws a [StateError] if ...
/// Similar to [anotherMethod()], but ...
```

## Markdown Support

dart doc processes Markdown formatting including:
- *Italic* and **bold** text
- Inline code with backticks
- Code blocks with triple backticks
- Links and headers

## Writing Guidelines

### Prefer Brevity

Be clear, precise, and terse. Eliminate unnecessary words.

### Avoid Abbreviations and Acronyms

Use full phrases instead of i.e., e.g., or specialized acronyms.

---

## Source

- **URL**: https://dart.dev/effective-dart/documentation
- **Fetched**: 2026-01-27
