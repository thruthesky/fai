# dart create - Command-Line Tool for Creating Dart Projects

## Overview

The `dart create` command enables developers to initialize Dart projects using predefined templates. This functionality is also accessible through integrated development environments. When executed, the command generates a project directory and automatically retrieves package dependencies unless the `--no-pub` flag is specified.

## Basic Project Creation

To generate a standard Dart project, use the following syntax:

```bash
$ dart create my_cli
```

This creates a directory named `my_cli` containing a simple console application using the default template.

## Template Selection

Different project types require different templates. Specify your preferred template using the `-t` or `--template` flag:

```bash
$ dart create -t web my_web_app
```

## Available Templates

| Template | Purpose |
|----------|---------|
| `cli` | Command-line application with argument parsing via `package:args` |
| `console` | Basic command-line application (default) |
| `package` | Shared Dart libraries package |
| `server-shelf` | Server application built with shelf |
| `web` | Web application using core Dart libraries |

All templates follow established package layout conventions.

## Additional Command Options

### Force Directory Creation

Override existing directories with the `--force` flag:

```bash
$ dart create --force <DIRECTORY>
```

### Display Help Information

Access comprehensive command documentation:

```bash
$ dart create --help
```

## Related Resources

For additional information about Dart command-line tools, consult the [Dart command-line tool documentation](/tools/dart-tool).
