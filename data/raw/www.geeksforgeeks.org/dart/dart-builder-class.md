# Dart Builder Class

## Overview

In Flutter development, the Builder class serves as a utility widget that provides a fresh BuildContext to its child widget. This is essential when you need to access a BuildContext that belongs to a different widget in the hierarchy.

## What is the Builder Class?

The main function of the Builder class is to build the child and return it. The Builder class passes a context to the child, It acts as a custom build function.

## Constructor Syntax

```dart
Builder({
  Key? key,
  required WidgetBuilder builder
})
```

**Important:** The builder parameter cannot be null.

## Available Methods

- **build(BuildContext context)** → Widget
- **createElement()** → StatelessElement
- **debugDescribeChildren()** → List<DiagnosticsNode>
- **debugFillProperties(DiagnosticPropertiesBuilder properties)** → void
- **noSuchMethod(Invocation invocation)** → dynamic
- **toString({DiagnosticLevel minLevel: DiagnosticLevel.info})** → String

## Common Use Case: Fixing Context Errors

A typical problem occurs when trying to display a SnackBar from within a Scaffold. The error arises because the context doesn't belong to the Scaffold widget.

### The Problem

```dart
body: Center(
  child: GestureDetector(
    onTap: () {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('GeeksforGeeks')),
      );
    },
    child: Container(/* ... */),
  ),
),
```

This fails because the context passed to the GestureDetector's onTap callback doesn't have access to the parent Scaffold.

### The Solution

Wrap the GestureDetector with a Builder widget:

```dart
body: Center(
  child: Builder(
    builder: (context) {
      return GestureDetector(
        onTap: () {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('GeeksforGeeks')),
          );
        },
        child: Container(/* ... */),
      );
    },
  ),
),
```

By wrapping with Builder, the new context now belongs to a widget that resides within the Scaffold hierarchy, allowing proper access to ScaffoldMessenger.

## Key Takeaway

The Builder widget solves context-related issues by creating a new scope where the BuildContext properly references the desired parent widget, enabling access to features like SnackBars and other Scaffold-dependent functionality.

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-builder-class/
- **Fetched**: 2026-01-27
