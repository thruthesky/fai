# Class Modifiers Reference

## Overview

This page documents the valid and invalid combinations of class modifiers in Dart, as specified in the language documentation.

## Valid Combinations

The following table shows permitted modifier combinations and their resulting capabilities:

| Declaration | Construct | Extend | Implement | Mix in | Exhaustive |
|---|---|---|---|---|---|
| `class` | Yes | Yes | Yes | No | No |
| `base class` | Yes | Yes | No | No | No |
| `interface class` | Yes | No | Yes | No | No |
| `final class` | Yes | No | No | No | No |
| `sealed class` | No | No | No | No | Yes |
| `abstract class` | No | Yes | Yes | No | No |
| `abstract base class` | No | Yes | No | No | No |
| `abstract interface class` | No | No | Yes | No | No |
| `abstract final class` | No | No | No | No | No |
| `mixin class` | Yes | Yes | Yes | Yes | No |
| `base mixin class` | Yes | Yes | No | Yes | No |
| `abstract mixin class` | No | Yes | Yes | Yes | No |
| `abstract base mixin class` | No | Yes | No | Yes | No |
| `mixin` | No | No | Yes | Yes | No |
| `base mixin` | No | No | No | Yes | No |

## Invalid Combinations

Certain modifier pairings are prohibited:

### Mutually Exclusive Modifiers

- `base`, `interface`, and `final` cannot be combined because they govern identical capabilities

### Redundant Pairings

- `sealed` with `abstract` - both prevent instantiation
- `sealed` with `base`, `interface`, or `final` - redundant restrictions
- `mixin` with `abstract` - both prevent instantiation
- `mixin` with `interface`, `final`, or `sealed` - these prevent the mixing capability

### Type Restrictions

- `enum` declarations cannot use any modifiers
- `extension type` declarations cannot use modifiers (they prevent extension/mixing and restrict implementation)

---

*This documentation reflects the official Dart language documentation.*
