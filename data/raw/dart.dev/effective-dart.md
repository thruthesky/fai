# Effective Dart

## Overview

Effective Dart provides best practices for building consistent, maintainable, and efficient Dart libraries. The guidance emphasizes two core principles:

1. **Consistency**: Code should look uniform unless differences reflect meaningful distinctions. When code stands out, it should do so for useful reasons.

2. **Brevity**: Dart includes features like string interpolation and initializing formals to help express intent simply. The goal is economical code, not cramped code.

## The Four Guide Categories

### Style Guide
Covers code layout, organization, identifier formatting (`camelCase`, `using_underscores`), and rules not handled by `dart format`.

### Documentation Guide
Explains documentation practices for both doc comments and regular code comments.

### Usage Guide
Teaches how to leverage language features for behavior implementation through statements and expressions.

### Design Guide
Provides guidance on API design, type signatures, and declarations for usable, consistent libraries.

## How to Read the Guidelines

Guidelines use standardized language to indicate their strength:

- **DO**: Follow always; valid reasons to deviate are nearly nonexistent
- **DON'T**: Avoid almost always; rarely justified exceptions exist
- **PREFER**: Generally follow; exceptional circumstances may warrant deviation
- **AVOID**: Generally skip; rare valid reasons for exceptions may exist
- **CONSIDER**: Optional practice depending on circumstances and preference

## Key Terminology

- **Library member**: Top-level field, getter, setter, or function
- **Class member**: Constructor, field, getter, setter, function, or operator within a class
- **Member**: Either library or class member
- **Property**: Field-like named construct (variable, getter, setter, or field)
- **Type**: Named type declaration (class, typedef, or enum)

## Summary of Guidelines by Category

### Style Rules
- Use `UpperCamelCase` for types and extensions
- Use `lowercase_with_underscores` for packages, directories, files, and import prefixes
- Use `lowerCamelCase` for other identifiers and constant names
- Capitalize acronyms longer than two letters
- Place `dart:` imports before others
- Place `package:` imports before relative imports
- Format code with `dart format`
- Use curly braces for all flow control statements
- Prefer lines of 80 characters or fewer

### Documentation Rules
- Use `///` for doc comments on members and types
- Write single-sentence summaries starting doc comments
- Use square brackets to reference in-scope identifiers
- Use prose to explain parameters, return values, and exceptions
- Prefer backtick fences for code blocks
- Avoid excessive markdown and HTML formatting
- Put doc comments before metadata annotations

### Usage Rules
- Use strings in `part of` directives
- Avoid importing from `src` directories of other packages
- Use relative import paths when appropriate
- Avoid explicit `null` initialization
- Use collection literals when possible
- Use adjacent strings for literal concatenation
- Prefer string interpolation over concatenation
- Use function declarations to bind functions to names
- Prefer async/await over raw futures
- Use `whereType()` to filter collections by type

### Design Rules
- Use consistent terminology throughout
- Avoid abbreviations
- Put the most descriptive noun last in names
- Make code read like sentences
- Use noun phrases for non-boolean properties
- Use non-imperative verb phrases for boolean properties
- Prefer imperative verbs for side-effect methods
- Avoid starting method names with "get"
- Use `to___()` for copying state to new objects
- Use `as___()` for different representations
- Make declarations private by default
- Use class modifiers to control extension and implementation
- Make fields and variables `final` when possible
- Use getters for property access; setters for property changes
- Annotate return types and parameter types on functions
- Avoid `dynamic` unless intentionally disabling type checking
- Use `Future<void>` for async members not producing values
- Avoid positional boolean parameters
- Override `hashCode` when overriding `==`

This comprehensive framework helps Dart developers write code that is predictable, readable, and maintainable across projects and teams.
