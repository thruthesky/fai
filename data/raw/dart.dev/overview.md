# Dart Overview - Complete Page Content

## Introduction

Dart is "a client-optimized language for developing fast apps on any platform." The language prioritizes productivity in multi-platform development with flexible execution across web, mobile, and desktop targets.

## Dart: The Language

Dart employs **static type checking** for type safety, though type annotations remain optional due to type inference capabilities. The language supports both strict typing and dynamic types for experimental code.

### Sound Null Safety

Dart implements built-in null safety, meaning variables cannot be null unless explicitly declared. This feature prevents null exceptions through static analysis while maintaining runtime enforcement.

### Code Example

The documentation includes a Monte Carlo estimation program demonstrating:
- Async/await patterns
- Stream generation
- Type inference
- Nullable and non-nullable types
- Generator functions

## Dart: The Libraries

### Core Libraries

Dart provides comprehensive built-in libraries:

- **dart:core** - fundamental types and collections
- **dart:async** - Future and Stream support for asynchronous programming
- **dart:math** - mathematical functions and random number generation
- **dart:convert** - data encoding/decoding (JSON, UTF-8)
- **dart:io** - file, socket, and HTTP operations
- **dart:ffi** - foreign function interfaces for C interoperability
- **dart:isolate** - concurrent programming through independent workers

### Additional Packages

Popular supplementary packages include: characters, intl, http, crypto, and markdown. The community provides thousands more covering XML, Windows integration, SQLite, and compression.

## Dart: The Platforms

### Native Platform

For mobile and desktop development, Dart includes:
- **JIT compiler** - fast development cycle with hot reload
- **AOT compiler** - produces optimized machine code for production

### Web Platform

For web applications, Dart compilers provide:
- Development mode with incremental compilation and hot reload
- Production JavaScript compilation with dead-code elimination
- WebAssembly (WasmGC) compilation for high performance

### The Runtime

The Dart runtime handles:
- Memory management through garbage collection
- Type system enforcement
- Isolate management for concurrent execution

## Learning Resources

Recommended starting points include:
- DartPad browser-based environment
- Official language tour documentation
- Command-line and server tutorials
- Online training courses
- API reference documentation
- Published programming books

---

**Current Version:** Dart 3.10.3
**License:** Content under Creative Commons Attribution 4.0; code samples under 3-Clause BSD License
