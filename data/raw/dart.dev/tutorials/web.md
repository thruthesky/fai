# Build a Web App with Dart

This guide walks through creating web applications using Dart, targeting developers comfortable with basic Dart concepts.

## Prerequisites

Before starting, ensure familiarity with Dart fundamentals by reviewing the Introduction to Dart documentation.

## Step 1: Install Dart

Download the Dart SDK from the official site, or install Flutter, which includes the complete Dart SDK. This provides all necessary tools for web development.

## Step 2: Choose Your Development Environment

**Command Line Option:**
Install the `webdev` package globally:
```bash
$ dart pub global activate webdev
```

**IDE Option:**
While optional, using an IDE with Dart integration is highly recommended. Several editors support Dart development.

The web template uses `package:web`, described as "Dart's powerful and concise web interop solution built for the modern web."

## Step 3: Create a Web App

**Command Line:**
```bash
$ dart create -t web quickstart
```

**IDE:**
Create a new project using the Bare-bones Web App template.

## Step 4: Run the App

**Command Line:**
```bash
$ cd quickstart
$ webdev serve
```

Access your app at `localhost:8080`. The development compiler builds incrementally after initial compilation, improving subsequent build speeds.

## Step 5: Customize Your App

Add this generator function to `web/main.dart`:

```dart
Iterable<String> thingsTodo() sync* {
  const actions = ['Walk', 'Wash', 'Feed'];
  const pets = ['cats', 'dogs'];

  for (final action in actions) {
    for (final pet in pets) {
      if (pet != 'cats' || action == 'Feed') {
        yield '$action the $pet';
      }
    }
  }
}
```

Create an element builder function:

```dart
HTMLLIElement newLI(String itemText) =>
  (document.createElement('li') as HTMLLIElement)..text = itemText;
```

Update the `main()` function:

```dart
void main() {
 final output = querySelector('#output');
 for (final item in thingsTodo()) {
   output?.appendChild(newLI(item));
 }
}
```

Optionally enhance styling in `web/styles.css`:

```css
#output {
  padding: 20px;
  text-align: left;
}
```

## Step 6: Debug Your Application

Use Dart DevTools to set breakpoints, inspect values, and step through code. See the debugging guide for setup instructions.

## Step 7: Build and Deploy

For production deployment outside development environments, build your app and deploy it. The web deployment documentation provides comprehensive guidance on this process.

## Additional Resources

- **Language Learning:** Language tour, core libraries, effective coding practices
- **Web Development:** JavaScript interoperability, web libraries, DOM introduction
- **Support:** Access community resources if assistance is needed
