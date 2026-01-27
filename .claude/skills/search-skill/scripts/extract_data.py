#!/usr/bin/env python3
"""
extract_data.py - 데이터 추출 및 정리 도구

Usage:
    python3 extract_data.py --input data.json --output result.md --format markdown
    python3 extract_data.py --input data.json --output result.csv --format csv
"""

import argparse
import json
import csv
import sys
from datetime import datetime
from typing import Any


def load_json(filepath: str) -> dict:
    """Load JSON data from file or stdin."""
    if filepath == '-':
        return json.load(sys.stdin)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_markdown_table(data: list[dict], columns: list[str] = None) -> str:
    """Convert list of dicts to markdown table."""
    if not data:
        return "No data available."
    
    if columns is None:
        columns = list(data[0].keys())
    
    # Header
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    
    # Rows
    rows = []
    for item in data:
        row_values = []
        for col in columns:
            val = item.get(col, "")
            # Escape pipe characters and truncate long values
            val_str = str(val).replace("|", "\\|")[:100]
            row_values.append(val_str)
        rows.append("| " + " | ".join(row_values) + " |")
    
    return "\n".join([header, separator] + rows)


def to_csv(data: list[dict], columns: list[str] = None) -> str:
    """Convert list of dicts to CSV string."""
    if not data:
        return ""
    
    if columns is None:
        columns = list(data[0].keys())
    
    output = []
    output.append(",".join(f'"{c}"' for c in columns))
    
    for item in data:
        row = []
        for col in columns:
            val = str(item.get(col, "")).replace('"', '""')
            row.append(f'"{val}"')
        output.append(",".join(row))
    
    return "\n".join(output)


def to_json(data: Any, pretty: bool = True) -> str:
    """Convert data to JSON string."""
    if pretty:
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps(data, ensure_ascii=False)


def generate_summary(data: list[dict]) -> dict:
    """Generate summary statistics for the data."""
    total = len(data)
    
    # Count valid images if 'valid' or 'image_valid' field exists
    valid_images = sum(1 for item in data if item.get('valid') or item.get('image_valid'))
    
    return {
        "total_items": total,
        "valid_images": valid_images,
        "invalid_images": total - valid_images,
        "generated_at": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Extract and format data")
    parser.add_argument("--input", "-i", default="-", help="Input JSON file (- for stdin)")
    parser.add_argument("--output", "-o", default="-", help="Output file (- for stdout)")
    parser.add_argument("--format", "-f", choices=["markdown", "csv", "json"], default="markdown")
    parser.add_argument("--columns", "-c", nargs="+", help="Columns to include")
    parser.add_argument("--summary", "-s", action="store_true", help="Include summary")
    
    args = parser.parse_args()
    
    # Load data
    data = load_json(args.input)
    
    # Handle both list and dict with 'results' key
    if isinstance(data, dict) and 'results' in data:
        items = data['results']
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    
    # Format output
    if args.format == "markdown":
        output = to_markdown_table(items, args.columns)
        if args.summary:
            summary = generate_summary(items)
            output = f"## Summary\n\n- Total: {summary['total_items']}\n- Valid: {summary['valid_images']}\n- Invalid: {summary['invalid_images']}\n\n## Data\n\n{output}"
    elif args.format == "csv":
        output = to_csv(items, args.columns)
    else:  # json
        if args.summary:
            output = to_json({"summary": generate_summary(items), "data": items})
        else:
            output = to_json(items)
    
    # Write output
    if args.output == "-":
        print(output)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
