# dartdevc: The Dart Development Compiler

## Overview

The Dart development compiler (dartdevc) is a specialized tool that compiles Dart code into JavaScript AMD modules, enabling web application development in contemporary browsers.

## Key Information

### Primary Function
"The Dart development compiler (_dartdevc_) compiles Dart as JavaScript AMD modules. These modules work for web app development in modern browsers."

### Important Note About Dart 2.18
Beginning with Dart 2.18, the `dartdevc` command-line utility was removed from the standard Dart package distribution. However, the underlying compiler functionality was preserved.

### Modern Workflow
Rather than using dartdevc directly through the command line, developers should leverage the `webdev serve` tool for compiling Dart to modular JavaScript. This approach provides two significant advantages:
- Faster compilation performance
- Simplified debugging experience

## Recommended Usage

The documentation emphasizes using [`webdev serve`](/tools/webdev#serve) as the contemporary method for compiling Dart code to modular JavaScript, representing the preferred development workflow for modern Dart web projects.

---

**Documentation Version:** Dart 3.10.3
**Last Updated:** September 15, 2025
