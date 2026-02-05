---
name: search-skill
description: Crawl Dart/Flutter official documentation sites and save as Markdown files in data/raw/ folder. IMPORTANT - All content must be fetched and saved in ENGLISH ONLY. Use this skill when (1) "/search <site>" or "/search <URL>" command is executed, (2) collecting official docs from dart.dev, docs.flutter.dev, etc., (3) Stage 1 data collection tasks.
user-invocable: true
---

# Dart/Flutter Documentation Collection Skill (Stage 1)

## Purpose

Handle Stage 1 data collection for the FAI project. Use WebFetch and WebSearch tools to crawl official documentation sites and save as Markdown files in `data/raw/` folder, preserving the URL path structure.

---

## ⚠️ CRITICAL: English-Only Data Rule ⚠️

**All data MUST be searched and saved in English only.**

| Rule | Description |
|------|-------------|
| **Search Language** | Always search and fetch content in **English** |
| **Save Language** | All Markdown files must be written in **English** |
| **WebFetch Prompt** | Use **English prompts** when fetching pages |
| **File Content** | No Korean or other languages in saved `.md` files |

**Why English only?**
- LLM training data should be consistent in language
- Official Dart/Flutter documentation is primarily in English
- Ensures uniform tokenization during model training

---

## Usage

```
/search <site>              # Crawl entire site
/search <URL>               # Fetch specific page only
```

**Examples:**
```
/search dart.dev            # Crawl all of dart.dev
/search docs.flutter.dev    # Crawl Flutter documentation
/search https://dart.dev/language/variables   # Specific page only
```

---

## URL → File Path Mapping Rules

| URL | Save Path |
|-----|-----------|
| `https://dart.dev/` | `data/raw/dart.dev/index.md` |
| `https://dart.dev/overview` | `data/raw/dart.dev/overview.md` |
| `https://dart.dev/language` | `data/raw/dart.dev/language.md` |
| `https://dart.dev/language/variables` | `data/raw/dart.dev/language/variables.md` |
| `https://docs.flutter.dev/` | `data/raw/docs.flutter.dev/index.md` |
| `https://docs.flutter.dev/ui/widgets` | `data/raw/docs.flutter.dev/ui/widgets.md` |
| `https://api.flutter.dev/flutter/widgets/StatefulWidget-class.html` | `data/raw/api.flutter.dev/flutter/widgets/StatefulWidget-class.md` |

**Conversion Rules:**
1. Remove `https://`
2. Use domain as top-level folder
3. Convert URL path to subfolders/files
4. Change `.html` extension to `.md`
5. If path ends with `/`, save as `index.md`

---

## Target Sites

| Site | Command | Save Path | Priority |
|------|---------|-----------|----------|
| dart.dev | `/search dart.dev` | `data/raw/dart.dev/` | 1 (Highest) |
| docs.flutter.dev | `/search docs.flutter.dev` | `data/raw/docs.flutter.dev/` | 2 |
| api.flutter.dev | `/search api.flutter.dev` | `data/raw/api.flutter.dev/` | 3 |
| api.dart.dev | `/search api.dart.dev` | `data/raw/api.dart.dev/` | 4 |
| pub.dev | `/search pub.dev` | `data/raw/pub.dev/` | 5 |

---

## Execution Procedure

### Step 1: Generate Seed URL List

Generate seed URL list for each domain using `extract_data.py --sitemap`:

```bash
python3 .claude/skills/search-skill/scripts/extract_data.py --sitemap dart.dev
```

### Step 2: Fetch Each Page with WebFetch

Fetch content for each URL using WebFetch tool.

**⚠️ IMPORTANT: Always use English prompts:**

```
WebFetch: https://dart.dev/language/variables
Prompt: "Extract the complete content of this page in Markdown format. Include all code examples and explanations in English."
```

### Step 3: Save and Update Index

**⚠️ CRITICAL: After fetching EACH page, you MUST:**

1. **Save the Markdown file** to `data/raw/<domain>/<path>.md`
2. **Immediately update `data/crawling-index.json`** with the crawl status

**DO NOT skip the index update!** The crawling index must be updated after every single page fetch to:
- Track which URLs have been collected
- Prevent duplicate crawling
- Enable resume from interruption
- Record timestamps and file paths

### Step 4: Save with extract_data.py

Save WebFetch results using `extract_data.py`:

```bash
# Save single page
python3 .claude/skills/search-skill/scripts/extract_data.py \
  --url "https://dart.dev/language/variables" \
  --content "WebFetch result..." \
  --output data/raw

# Or batch process with JSON
echo '[{"url": "...", "content": "..."}]' | \
  python3 .claude/skills/search-skill/scripts/extract_data.py --output data/raw
```

### Step 4: Check Collection Status

```bash
# Overall status
python3 .claude/skills/search-skill/scripts/extract_data.py --status --output data/raw

# Status for specific domain
python3 .claude/skills/search-skill/scripts/extract_data.py --status --domain dart.dev --output data/raw
```

---

## extract_data.py Script

### Main Options

| Option | Description |
|--------|-------------|
| `--url, -u` | URL of page to save |
| `--content, -c` | Page content (Markdown) |
| `--title, -t` | Page title (optional) |
| `--input, -i` | Input JSON file (- for stdin) |
| `--output, -o` | Output base directory (default: data/raw) |
| `--sitemap, -s` | Generate seed URL list for domain |
| `--status` | Display collection status |
| `--domain, -d` | Filter by specific domain |
| `--links` | Extract links from content |
| `--json` | Output in JSON format |

### Usage Examples

```bash
# Generate seed URL list
python3 extract_data.py --sitemap dart.dev

# Save single page
python3 extract_data.py --url "https://dart.dev/language" --content "..." --output data/raw

# JSON input from stdin
echo '{"url": "...", "content": "..."}' | python3 extract_data.py --output data/raw

# Collection status (including uncollected URLs)
python3 extract_data.py --status --domain dart.dev --json --output data/raw
```

---

## Crawling Index (data/crawling-index.json)

Manages crawling state and logs to prevent duplicate crawling and support re-crawling when needed.

### File Structure

```json
{
  "version": "1.0",
  "last_updated": "2024-01-27T10:30:00Z",
  "settings": {
    "max_age_days": 30,
    "auto_refresh": true
  },
  "urls": {
    "https://dart.dev/language/variables": {
      "status": "completed",
      "file_path": "data/raw/dart.dev/language/variables.md",
      "first_crawled": "2024-01-25T08:00:00Z",
      "last_crawled": "2024-01-27T10:30:00Z",
      "crawl_count": 2,
      "content_hash": "a1b2c3d4e5f6...",
      "file_size": 4523,
      "error": null
    }
  },
  "statistics": {
    "total_urls": 150,
    "completed": 120,
    "failed": 5,
    "pending": 25,
    "total_size_bytes": 2458000
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `status` | `pending`, `in_progress`, `completed`, `failed`, `stale` |
| `file_path` | Path to saved Markdown file |
| `first_crawled` | First crawl time (ISO 8601) |
| `last_crawled` | Last crawl time (ISO 8601) |
| `crawl_count` | Total crawl count |
| `content_hash` | MD5 hash of content (for change detection) |
| `file_size` | Saved file size (bytes) |
| `error` | Error message on failure |

---

## Saved File Format

**⚠️ All content must be in English:**

```markdown
# [Page Title]

[Original document content converted to Markdown - IN ENGLISH]

## Overview
...

## Details
...

## Code Examples
```dart
// Example code from the page
void main() {
  print('Hello, Dart!');
}
```

---

## Source

- **URL**: https://dart.dev/language/variables
- **Fetched**: 2024-01-27
```

---

## Crawling Workflow (Claude Execution)

1. **Initialize Index**: Create index with seed URLs on first crawl
   ```bash
   python3 extract_data.py --init-index --domain dart.dev
   ```

2. **Check Crawl Targets**: Query pending/stale URL list
   ```bash
   python3 extract_data.py --check-index --domain dart.dev
   ```

3. **Iterative Crawling**: For each URL:
   - Check status in index (skip if completed and within max_age)
   - Fetch content with WebFetch **(English prompt required)**
   - Save Markdown file to `data/raw/`
   - **⚠️ IMMEDIATELY update `data/crawling-index.json`** (DO NOT batch updates!)

4. **Link Discovery**: When new links found in WebFetch results:
   - Add to index as pending if not exists
   - Continue collection

5. **Check Progress**: Review collection status via index statistics
   ```bash
   python3 extract_data.py --status --domain dart.dev
   ```

6. **Retry Failed URLs**: Check failed URL list and retry
   ```bash
   python3 extract_data.py --list-failed --domain dart.dev
   ```

---

## Crawling Guidelines

1. **Respect robots.txt**: Check each site's crawling policy
2. **Request Interval**: Use appropriate delays to prevent server overload
3. **Prevent Duplicates**: Skip already collected files (overwrite option available)
4. **Error Handling**: Log failed URLs and implement retry logic
5. **English Only**: Always fetch and save content in English

---

## Supported Domain URL Lists

### dart.dev (60+ URLs)

Main sections: `/language/*`, `/libraries/*`, `/tutorials/*`, `/effective-dart/*`, `/tools/*`

### docs.flutter.dev (45+ URLs)

Main sections: `/get-started/*`, `/ui/*`, `/development/*`, `/testing/*`, `/deployment/*`

### api.flutter.dev (30+ URLs)

Main classes: `StatelessWidget`, `StatefulWidget`, `Container`, `Text`, `Row`, `Column`, Material/Cupertino widgets

### api.dart.dev (15+ URLs)

Main libraries: `dart-core`, `dart-async`, `dart-collection`, `dart-convert`, `dart-io`

### pub.dev (25+ URLs)

Popular packages: `provider`, `bloc`, `riverpod`, `dio`, `hive`, `firebase_*`, etc.

---

## Related Skills

- **fai-skill**: Stage 2 (preprocessing) and overall project management
