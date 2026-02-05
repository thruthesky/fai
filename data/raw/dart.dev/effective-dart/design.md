# Effective Dart: Design

This comprehensive guide addresses how to write consistent, usable APIs for Dart libraries, organized into nine key sections.

## Names

Effective naming requires using consistent terminology throughout your codebase and leveraging existing conventions. Abbreviations should generally be avoided unless they're more recognizable than full terms.

**Structural guidance**: Place the most descriptive noun last in names (e.g., `ConversionSink`), and structure code to read naturally as sentences.

**Boolean naming**: Use non-imperative verb phrases like `isEnabled`, `hasElements`, or `canClose`. Prefer positive phrasing over negation—`if (socket.isConnected)` reads better than `if (!socket.isDisconnected)`.

**Method naming patterns**:
- Imperative verbs for side-effect operations: `list.add()`, `window.refresh()`
- Noun phrases for value-returning operations: `list.elementAt(3)`, `list.firstWhere()`
- Conversion methods use `to___()` for new copies and `as___()` for views backed by originals

Avoid prefixing method names with "get"—either convert to a property or use a more descriptive verb.

## Libraries

Make declarations private by default using the underscore prefix. This narrows your public interface and helps maintainers track unused code. Dart permits multiple related classes in a single library, enabling "friend" patterns through library-level privacy.

## Classes and Mixins

Avoid one-member abstract classes when functions suffice. Don't define classes containing only static members—use top-level declarations or libraries instead.

**Subclassing constraints**: Don't extend classes not designed for extension. Use class modifiers like `final`, `base`, or `sealed` to explicitly control extensibility and implementation. This protects library maintainers when evolving APIs.

Prefer pure `mixin` or `class` declarations over `mixin class` for new code.

## Constructors

Mark constructors `const` when possible—this enables using instances in constant contexts and signals immutability.

## Members

Make fields and top-level variables `final` by default. Use getters for "property-like" operations that don't modify state, take no arguments, and remain idempotent. Use setters for operations that conceptually change properties. Always provide a corresponding getter for any setter.

Avoid returning `this` to enable fluent interfaces—use method cascades instead: `StringBuffer()..write('one')..write('two')`.

## Types

**Variables**: Annotate variables without initializers, fields where type isn't obvious, and all function declaration parameters and returns.

**Inference**: Omit types on initialized local variables and inferred generic invocations. Write explicit types on generic invocations that aren't inferred.

**Dynamic usage**: Explicitly annotate `dynamic` rather than relying on implicit inference failure.

**Function types**: Prefer complete function signatures over bare `Function`. Use inline function types instead of typedefs in most cases.

## Parameters

Avoid positional boolean parameters—use named parameters instead. Don't use optional positional parameters when users might skip earlier ones. Use inclusive-start, exclusive-end conventions for ranges.

## Equality

Override `hashCode` whenever overriding `==`. Ensure the `==` operator obeys mathematical equality rules. Avoid custom equality for mutable classes, and don't make the equality parameter nullable.

---

**Key principle**: Design APIs that communicate intent clearly through naming conventions, minimize public surface area, and leverage Dart's type system to guide safe, maintainable code.
