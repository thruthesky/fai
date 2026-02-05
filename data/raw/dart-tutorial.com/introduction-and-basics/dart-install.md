# Install Dart

## Dart Installation

Multiple installation methods are available for various operating systems, including Windows, Mac, and Linux. Additionally, you can run Dart directly in a web browser.

## Requirements

- Dart SDK
- A code editor such as VS Code or IntelliJ

## Dart Windows Installation

### Steps:

1. Download Dart SDK from the [official archive](https://dart.dev/get-dart/archive)
2. Transfer the **dart-sdk** folder to your C drive
3. Add **C:\dart-sdk\bin** to your environment variables (refer to video tutorial for clarity)
4. Open command prompt and run **`dart --version`** to verify installation
5. Install [VS Code](https://code.visualstudio.com/download) and add the Dart extension

**Note:** "Dart SDK provides the tools to compile and run dart program."

## Dart Mac Installation

1. Install Homebrew from [here](https://brew.sh/)
2. Execute `brew tap dart-lang/dart` in terminal
3. Execute `brew install dart` in terminal
4. Consult video tutorial if installation issues arise

### Homebrew Installation Commands

Install Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Set Homebrew path:
```bash
export PATH=/opt/homebrew/bin:$PATH
```

## Dart Linux Installation

Copy and paste these commands into your terminal:

```bash
sudo apt-get update
sudo apt-get install apt-transport-https
wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/dart.gpg
echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' | sudo tee /etc/apt/sources.list.d/dart_stable.list
```

Install Dart:
```bash
sudo apt-get update
sudo apt-get install dart
```

Set the Dart path:
```bash
export PATH="$PATH:/usr/lib/dart/bin"
```

## Check Dart Installation

Run **`dart --version`** in your command prompt. Successful installation displays version information.

## Useful Commands

| Command | Purpose |
|---------|---------|
| `dart --help` | Display all available commands |
| `dart filename.dart` | Execute a Dart file |
| `dart create` | Initialize a new Dart project |
| `dart fix` | Update project to current syntax standards |
| `dart compile exe bin/dart.dart` | Generate executable file |
| `dart compile js bin/dart.dart` | Compile to JavaScript for Node.js execution |

## Run Dart On Web

[DartPad](https://dartpad.dev) is a browser-based tool for writing and executing Dart code without local installation.

## Additional Resources

- [Official Dart Installation Guide](https://dart.dev/get-dart)

## Mobile Execution

"Yes, you can use DartPad to run simple dart programs from your phone without installing any software." However, larger projects are better suited for local development environments.

---

## Source

- **URL**: https://dart-tutorial.com/introduction-and-basics/dart-install/
- **Fetched**: 2026-01-27
