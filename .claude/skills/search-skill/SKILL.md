---
name: search-skill
description: Crawl Dart/Flutter official documentation sites and save directly to PostgreSQL (Supabase). Documents and crawling state are stored in the crawled_documents table. IMPORTANT - All content must be fetched and saved in ENGLISH ONLY. Use this skill when (1) "/search <site>" or "/search <URL>" command is executed, (2) collecting official docs from dart.dev, docs.flutter.dev, etc., (3) Stage 1 data collection tasks.
user-invocable: true
---

# Dart/Flutter Documentation Collection Skill (Stage 1)

## Purpose

Handle Stage 1 data collection for the FAI project. Use WebFetch and WebSearch tools to crawl official documentation sites and save directly to PostgreSQL (Supabase) `crawled_documents` table.

**Requirements:** `.environments` file must exist in the project root with Supabase connection info.

---

## ⚠️ CRITICAL: English-Only Data Rule ⚠️

**All data MUST be searched and saved in English only.**

| Rule | Description |
|------|-------------|
| **Search Language** | Always search and fetch content in **English** |
| **Save Language** | All content must be written in **English** |
| **WebFetch Prompt** | Use **English prompts** when fetching pages |
| **DB Content** | No Korean or other languages in saved content |

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

## Target Sites

| Site | Command | Priority |
|------|---------|----------|
| dart.dev | `/search dart.dev` | 1 (Highest) |
| docs.flutter.dev | `/search docs.flutter.dev` | 2 |
| api.flutter.dev | `/search api.flutter.dev` | 3 |
| api.dart.dev | `/search api.dart.dev` | 4 |
| pub.dev | `/search pub.dev` | 5 |

---

## Execution Procedure

### Step 1: Generate Seed URL List

Generate seed URL list for each domain using `extract_data.py --sitemap`:

```bash
uv run python .claude/skills/search-skill/scripts/extract_data.py --sitemap dart.dev
```

### Step 2: Fetch Each Page with WebFetch

Fetch content for each URL using WebFetch tool.

**⚠️ IMPORTANT: Always use English prompts:**

```
WebFetch: https://dart.dev/language/variables
Prompt: "Extract the complete content of this page in Markdown format. Include all code examples and explanations in English."
```

### Step 3: Save to PostgreSQL

Save WebFetch results directly to DB using `extract_data.py`:

```bash
# Save single page to DB
uv run python .claude/skills/search-skill/scripts/extract_data.py \
  --url "https://dart.dev/language/variables" \
  --content "WebFetch result..."

# Or batch process with JSON
echo '[{"url": "...", "content": "..."}]' | \
  uv run python .claude/skills/search-skill/scripts/extract_data.py
```

### Step 4: Check Collection Status

```bash
# Overall status (DB query)
uv run python .claude/skills/search-skill/scripts/extract_data.py --status

# Status for specific domain
uv run python .claude/skills/search-skill/scripts/extract_data.py --status --domain dart.dev
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
| `--sitemap, -s` | Generate seed URL list for domain |
| `--status` | Display collection status (DB query) |
| `--domain, -d` | Filter by specific domain |
| `--links` | Extract links from content |
| `--json` | Output in JSON format |

### Usage Examples

```bash
# Generate seed URL list (no DB connection needed)
uv run python extract_data.py --sitemap dart.dev

# Save single page to DB
uv run python extract_data.py --url "https://dart.dev/language" --content "..."

# JSON input from stdin
echo '{"url": "...", "content": "..."}' | uv run python extract_data.py

# Collection status (DB query, including uncollected URLs)
uv run python extract_data.py --status --domain dart.dev --json
```

---

## DB Storage (crawled_documents Table)

Documents and crawling state are stored in a single `crawled_documents` table in PostgreSQL (Supabase).

### Table Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `url` | TEXT (UNIQUE) | Original URL |
| `domain` | TEXT | Domain (e.g., dart.dev) |
| `url_path` | TEXT | URL path (e.g., /language/variables) |
| `title` | TEXT | Page title |
| `content` | TEXT | Markdown content |
| `content_hash` | TEXT | MD5 hash (change detection) |
| `content_size` | INTEGER | Content size in bytes |
| `status` | TEXT | pending, completed, failed |
| `error` | TEXT | Error message on failure |
| `first_crawled` | TIMESTAMP | First crawl time |
| `last_crawled` | TIMESTAMP | Last crawl time |
| `crawl_count` | INTEGER | Total crawl count |

### ORM Model

Defined in `distributed/server/models.py` as `CrawledDocument` class.

---

## Crawling Workflow (Claude Execution)

1. **Generate Seed URLs**: Create URL list for target domain
   ```bash
   uv run python extract_data.py --sitemap dart.dev
   ```

2. **Check Already Collected**: Query DB for completed URLs
   ```bash
   uv run python extract_data.py --status --domain dart.dev
   ```

3. **Iterative Crawling**: For each URL:
   - Check status in DB (skip if already completed)
   - Fetch content with WebFetch **(English prompt required)**
   - Save to DB via extract_data.py (UPSERT: insert or update)

4. **Link Discovery**: When new links found in WebFetch results:
   - Check if URL already exists in DB
   - If not, fetch and save

5. **Check Progress**: Review collection status via DB statistics
   ```bash
   uv run python extract_data.py --status --domain dart.dev
   ```

---

## Crawling Guidelines

1. **Respect robots.txt**: Check each site's crawling policy
2. **Request Interval**: Use appropriate delays to prevent server overload
3. **Prevent Duplicates**: DB UNIQUE constraint on URL prevents duplicates
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
