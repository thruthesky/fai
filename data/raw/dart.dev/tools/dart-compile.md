# Dart Compile Command Documentation

## Overview

The `dart compile` command compiles Dart source code to various target platforms. As stated in the documentation, "Use the `dart compile` command to compile a Dart program to a target platform."

**Important Note:** If your package uses build hooks, you must use `dart build` instead, as `dart compile exe` and `dart compile aot-snapshot` do not run build hooks.

## Basic Usage Examples

Compile to a self-contained executable:
```bash
$ dart compile exe bin/myapp.dart
Generated: /Users/me/myapp/bin/myapp.exe
```

Compile to an AOT module, then run it:
```bash
$ dart compile aot-snapshot bin/myapp.dart
Generated: /Users/me/myapp/bin/myapp.aot
$ dartaotruntime bin/myapp.aot
```

Specify custom output location:
```bash
$ dart compile exe bin/myapp.dart -o bin/runme
```

## Subcommands

| Subcommand | Output Type | Purpose |
|-----------|------------|---------|
| `exe` | Self-contained executable | "A standalone, architecture-specific executable file containing the source code compiled to machine code and a small Dart runtime" |
| `aot-snapshot` | AOT module | Architecture-specific compiled code without runtime |
| `jit-snapshot` | JIT module | "An architecture-specific file with an intermediate representation of all source code, plus an optimized representation" |
| `kernel` | Kernel module | "A portable, intermediate representation of the source code" |
| `js` | JavaScript | Deployable JavaScript file |
| `wasm` | WebAssembly | Currently under development |

## Self-Contained Executables (exe)

The `exe` subcommand produces native machine code for Windows, macOS, or Linux with a small integrated runtime.

### Cross-Compilation

Cross-compilation to Linux is supported from 64-bit host systems (Linux, macOS, Windows). Use these flags:

```bash
dart compile exe \
  --target-os=linux \
  --target-arch=x64 \
  hello.dart
```

Target architectures available:
- `arm`: 32-bit ARM
- `arm64`: 64-bit ARM
- `riscv64`: 64-bit RISC-V
- `x64`: x86-64

### Known Limitations

- No support for `dart:mirrors` and `dart:developer`
- Cross-compilation limited to Linux targets

## AOT Modules (aot-snapshot)

AOT modules are useful for reducing disk space when distributing multiple command-line applications. They are architecture-specific. Run them using:

```bash
$ dartaotruntime bin/myapp.aot
```

Cross-compilation support matches the `exe` subcommand capabilities.

## JIT Modules (jit-snapshot)

JIT modules contain parsed classes and compiled code from a training run:

```bash
$ dart compile jit-snapshot bin/myapp.dart
$ dart run bin/myapp.jit
```

These are architecture-specific, unlike kernel modules.

## Portable Modules (kernel)

The `kernel` subcommand creates portable files runnable on all operating systems:

```bash
$ dart compile kernel bin/myapp.dart
Compiling bin/myapp.dart to kernel file bin/myapp.dill.
$ dart run bin/myapp.dill
```

These have reduced startup time compared to raw Dart code but slower startup than AOT formats.

## JavaScript Compilation (js)

The `dart compile js` command compiles to deployable JavaScript. The documentation recommends using the `webdev` tool instead for most web applications.

### Basic Options

- `-o <file>`: Specify output file
- `--enable-asserts`: Enable assertion checking
- `-O{0|1|2|3|4}`: Optimization levels (0=minimal, 2=safe, 3+=aggressive)
- `--no-source-maps`: Skip source map generation

### Optimization Levels

- **-O2**: "Enables `-O1` optimizations, plus additional ones (such as minification) that respect the language semantics and are safe for all programs"
- **-O3**: Omits implicit type checks; test with -O2 first
- **-O4**: More aggressive than -O3; test edge cases before use

### Path and Environment Options

- `--packages=<path>`: Package resolution configuration file
- `-D<flag>=<value>`: Environment declarations accessible via `String.fromEnvironment()` and similar functions
- `--version`: Display version information

### Display Options

- `--suppress-warnings`: Hide warnings
- `--suppress-hints`: Hide hints
- `--terse`: Diagnostic messages without solutions
- `-v` or `--verbose`: Detailed output
- `--enable-diagnostic-colors`: Add color to messages

### Analysis Options

- `--fatal-warnings`: Treat warnings as errors
- `--show-package-warnings`: Display package-generated diagnostics
- `--csp`: Disable dynamic code generation for CSP compliance
- `--dump-info`: Generate `.info.json` file for code analysis

### Example Web Compilation

```bash
$ dart compile js -O2 -o out/main.js web/main.dart
```

### Best Practices for Efficient Code

The documentation advises developers to:
- Avoid `Function.apply()`
- Don't override `noSuchMethod()`
- Avoid assigning `null` to variables
- Maintain consistent argument types

The compiler automatically performs tree shaking to eliminate unused code, so developers should import needed libraries without concern about bundle size.

## Additional Resources

For help with all options, use the `-hv` flag with any subcommand. The documentation mentions that "You don't need to compile Dart programs before running them. Instead, you can use the `dart run` command, which uses the Dart VM's JIT (just-in-time) compiler."
