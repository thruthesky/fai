# Dart SDK Installation Guide

## Overview

To do a lot of interesting programming stuff using the Dart programming language, we have to install the Dart SDK. The SDK comes as a pre-compiled package requiring only download and extraction.

---

## Windows Installation

### Method 1: Manual Download

**Step 1: Download Dart SDK**
- Visit the [Dart SDK archive page](https://dart.dev/get-dart/archive)
- Select the Windows 64-Bit architecture version
- Download the ZIP file

**Step 2: Extract Files**
Extract the downloaded ZIP file to your preferred location. The folder structure will contain a `bin` directory with executable files.

**Step 3: Run Dart**
Navigate to the `bin` folder and open Command Prompt there, then type `dart` to verify installation.

### Method 2: Chocolatey Installation

For users with Chocolatey package manager installed:

```
C:\> choco install dart-sdk
```

To update:
```
C:\> choco upgrade dart-sdk
```

### Setting Environment Variables

To use Dart from any location in your system:
1. Open Environment Variables from Advanced System Settings
2. Add the Dart SDK bin folder path to System Variables
3. This allows running `dart` commands from any Command Prompt window

---

## Linux Installation

### Step 1: Update Package Manager
```bash
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install apt-transport-https
```

### Step 2: Add Dart Repository and GPG Keys
```bash
$ sudo sh -c 'wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -'
$ sudo sh -c 'wget -qO- https://storage.googleapis.com/download.dartlang.org/linux/debian/dart_stable.list > /etc/apt/sources.list.d/dart_stable.list'
```

### Step 3: Install Dart
```bash
$ sudo apt-get update
$ sudo apt-get install dart
```

### Step 4: Configure PATH
```bash
$ export PATH="$PATH:/usr/lib/dart/bin"
$ source ~/.bashrc
$ dart --version
```

---

## macOS Installation

### Step 1: Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Dart
```bash
$ brew tap dart-lang/dart
$ brew install dart
```

### Step 3: Update Dart
```bash
$ brew upgrade dart
```

### Step 4: Verify Installation
```bash
$ dart --version
```

---

## Key Notes

- Dart SDK is a pre-compiled version so we have to download and extract it only
- Flutter developers have Dart bundled with the Flutter SDK, eliminating the need for a separate installation
- Setting environment variables enables running Dart commands from any terminal/command prompt location
- The installation process varies by operating system but achieves the same outcome

---

## Source

- **URL**: https://www.geeksforgeeks.org/dart/dart-sdk-installation/
- **Fetched**: 2026-01-27
