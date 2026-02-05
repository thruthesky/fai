# Dart Keywords Reference

## Overview

This page documents the reserved words in the Dart programming language that developers cannot use as standard identifiers. The complete list includes 60+ keywords with varying restriction levels.

## Key Points About Dart Keywords

### Restriction Categories

Keywords fall into three usage categories:

1. **Context-dependent keywords** (like `await` and `yield`) - Can serve as identifiers in certain contexts
2. **Type-restricted keywords** (marked with superscript 2) - Cannot name types, extensions, or import prefixes, but usable elsewhere
3. **Unrestricted keywords** (marked with superscript 3) - Freely used as identifiers without limitation

### Complete Keyword List

The language reserves words spanning several functional areas:

- **Control flow**: `if`, `else`, `switch`, `case`, `default`, `do`, `while`, `for`, `break`, `continue`
- **Error handling**: `try`, `catch`, `finally`, `throw`, `rethrow`, `assert`, `on`
- **Classes & objects**: `class`, `abstract`, `extends`, `implements`, `mixin`, `enum`, `factory`
- **Access modifiers**: `static`, `final`, `const`, `late`, `required`, `covariant`
- **Functions**: `Function`, `async`, `sync`, `yield`, `get`, `set`, `operator`, `external`
- **Libraries**: `import`, `export`, `library`, `part`, `deferred`, `show`, `hide`
- **Type system**: `dynamic`, `as`, `is`, `typedef`, `type`
- **Other**: `true`, `false`, `null`, `this`, `super`, `new`, `var`, `void`, `base`, `interface`, `sealed`

## Best Practices

While some keywords technically work as identifiers in certain contexts, using them that way creates confusion for other developers and should be avoided as a matter of code quality and readability.
