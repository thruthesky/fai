# Dart URIs - Complete Guide

## Overview

The `Uri` class in Dart provides methods to encode and decode strings for use in URIs (also known as URLs). It handles special characters unique to URIs like `&` and `=`, while also parsing and exposing URI components such as host, port, and scheme.

## 1. Encoding and Decoding Fully Qualified URIs

### Methods: `encodeFull()` and `decodeFull()`

These methods encode and decode characters **except** those with special meaning in URIs (such as `/`, `:`, `&`, `#`).

**Example:**

```dart
void main(){
  var uri = 'https://example.org/api?foo=some message';

  var encoded = Uri.encodeFull(uri);
  assert(encoded == 'https://example.org/api?foo=some%20message');

  var decoded = Uri.decodeFull(encoded);
  print(uri == decoded);
}
```

**Output:** `true`

---

## 2. Encoding and Decoding URI Components

### Methods: `encodeComponent()` and `decodeComponent()`

These methods encode and decode **all** characters with special meaning in URIs, including `/`, `&`, and `:`.

**Example:**

```dart
void main() {
  var uri = 'https://example.org/api?foo=some message';

  var encoded = Uri.encodeComponent(uri);
  assert(encoded == 'https%3A%2F%2Fexample.org%2Fapi%3Ffoo%3Dsome%20message');

  var decoded = Uri.decodeComponent(encoded);
  print(uri == decoded);
}
```

**Output:** `true`

---

## 3. Parsing URIs

### Method: `parse()`

Use the static `parse()` method to create a URI object from a string. Access individual components using properties like `scheme`, `host`, `path`, and `fragment`.

**Example:**

```dart
void main() {
  var uri = Uri.parse('https://example.org:8080/foo/bar#frag');

  assert(uri.scheme == 'https');
  assert(uri.host == 'example.org');
  assert(uri.path == '/foo/bar');
  assert(uri.fragment == 'frag');
  print(uri.origin == 'https://example.org:8080');
}
```

**Output:** `true`

---

## 4. Building URIs

### Constructor: `Uri()`

Construct a URI from individual components using the `Uri()` constructor with named parameters.

**Example:**

```dart
void main() {
  var uri = Uri(
    scheme: 'https',
    host: 'example.org',
    path: '/foo/bar',
    fragment: 'frag',
  );
  print(uri.toString() == 'https://example.org/foo/bar#frag');
}
```

**Output:** `true`

---

## Key Takeaways

- Use **`encodeFull()`/`decodeFull()`** for preserving URI structure while encoding spaces and special text
- Use **`encodeComponent()`/`decodeComponent()`** when encoding complete URI strings as data
- Use **`parse()`** to extract and examine URI components
- Use the **`Uri()` constructor** to programmatically build URIs from parts

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-uris/
- **Fetched**: 2026-01-27
