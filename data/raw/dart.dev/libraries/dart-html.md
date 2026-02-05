# dart:html Library Documentation

## Overview

The `dart:html` library enables Dart developers to program browsers and manipulate Document Object Model (DOM) elements. However, the library is now deprecated in favor of `dart:js_interop` and `package:web`.

**Key Note:** "Only web apps can use dart:html, not command-line apps."

## Core Concepts

### DOM Structure

The DOM consists of hierarchical components:

- **Window**: Represents the browser window, providing access to APIs like IndexedDB and requestAnimationFrame
- **Document**: Contains the currently loaded page content
- **Element**: Individual components within the document
- **Node**: Basic building blocks, which can be elements, attributes, text, or comments

## DOM Manipulation

### Finding Elements

Developers use `querySelector()` and `querySelectorAll()` with CSS selectors:

```dart
Element idElement = querySelector('#an-id')!;
Element classElement = querySelector('.a-class')!;
List<Element> divElements = querySelectorAll('div');
List<Element> textInputElements = querySelectorAll('input[type="text"]');
```

The first function returns a single element; the second returns a collection.

### Modifying Elements

Properties allow state changes on elements:

```dart
var anchor = querySelector('#example') as AnchorElement;
anchor.href = 'https://dart.dev';
```

For bulk modifications, iterate through matched elements. The `hidden` property controls visibility similarly to CSS `display: none`.

### Creating Elements

```dart
var elem = ParagraphElement();
elem.text = 'Creating is easy!';

var elem2 = Element.html('<p>Creating <em>is</em> easy!</p>');
document.body!.children.add(elem2);
```

HTML parsing automatically creates child elements.

### Node Operations

Manipulate node collections through the `nodes` property:

```dart
querySelector('#inputs')!.nodes.add(elem);
querySelector('#status')!.replaceWith(elem);
querySelector('#expendable')?.remove();
```

## CSS and Styling

### Classes

Modify CSS classes through a list interface:

```dart
var elem = querySelector('#message')!;
elem.classes.add('warning');
```

### Direct Styles

Apply inline styles using cascading syntax:

```dart
var message = DivElement()
  ..id = 'message2'
  ..text = 'Please subscribe to the Dart mailing list.';

message.style
  ..fontWeight = 'bold'
  ..fontSize = '3em';
```

## Event Handling

Register event listeners using the `on[Event].listen()` pattern:

```dart
querySelector('#submitInfo')!.onClick.listen((e) {
  submitData();
});

document.body!.onClick.listen((e) {
  final clickedElem = e.target;
});
```

Common events include: change, blur, keyDown, keyUp, mouseDown, and mouseUp.

## HTTP Requests

"Use a higher-level library like `package:http`" rather than directly using `HttpRequest` from dart:html, as it remains platform-dependent.

## WebSockets

### Creating and Sending

```dart
var ws = WebSocket('ws://echo.websocket.org');
ws.send('Hello from Dart!');
```

### Receiving Data

```dart
ws.onMessage.listen((MessageEvent e) {
  print('Received message: ${e.data}');
});
```

### Event Handling

Manage open, close, error, and message events through listener registration to enable reconnection logic and error recovery.

## Deprecation Notice

"The `dart:html` library is deprecated. Instead, use `dart:js_interop` and `package:web`."

For larger applications, consider Flutter for web as an alternative approach.
