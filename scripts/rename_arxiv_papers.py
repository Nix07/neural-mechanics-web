#!/usr/bin/env python3
"""
Rename arxiv papers to uniform format:
firstauthorlastname-publicationyear-shortenedtitle-arxivid.pdf
"""

import os
import re
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"

def extract_arxiv_id(filename):
    """Extract arxiv ID from filename like arxiv_1234.56789_*.pdf"""
    match = re.search(r'arxiv_(\d+\.\d+)', filename)
    if match:
        return match.group(1)
    return None

def fetch_arxiv_metadata(arxiv_id):
    """Fetch metadata from arxiv API"""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_data = response.read().decode('utf-8')

        # Parse XML
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        # Get title
        title_elem = entry.find('atom:title', ns)
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

        # Get first author
        authors = entry.findall('atom:author', ns)
        if authors:
            first_author = authors[0].find('atom:name', ns)
            author_name = first_author.text.strip() if first_author is not None else ""
        else:
            author_name = ""

        # Get publication date
        published = entry.find('atom:published', ns)
        pub_date = published.text[:4] if published is not None else ""  # Extract year

        return {
            'title': title,
            'author': author_name,
            'year': pub_date,
            'arxiv_id': arxiv_id
        }
    except Exception as e:
        print(f"Error fetching {arxiv_id}: {e}")
        return None

def get_author_lastname(author_name):
    """Extract last name from author name"""
    if not author_name:
        return "unknown"
    parts = author_name.split()
    if len(parts) >= 1:
        return parts[-1].lower()
    return "unknown"

def shorten_title(title, max_words=4):
    """Create a shortened title slug"""
    if not title:
        return "untitled"
    # Remove special characters and lowercase
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    words = title.split()
    # Take first few significant words (skip common words)
    skip_words = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'and', 'or', 'to', 'with', 'by', 'from', 'as', 'is', 'are', 'at'}
    significant = [w for w in words if w.lower() not in skip_words][:max_words]
    if not significant:
        significant = words[:max_words]
    return '-'.join(w.lower() for w in significant)

def main():
    # Find all arxiv PDFs
    arxiv_files = []
    for f in os.listdir(PAPERS_DIR):
        if f.startswith('arxiv_') and f.endswith('.pdf'):
            arxiv_id = extract_arxiv_id(f)
            if arxiv_id:
                arxiv_files.append((f, arxiv_id))

    print(f"Found {len(arxiv_files)} arxiv papers to process\n")

    renames = []

    for old_name, arxiv_id in sorted(arxiv_files, key=lambda x: x[1]):
        print(f"Fetching metadata for {arxiv_id}...")
        metadata = fetch_arxiv_metadata(arxiv_id)
        time.sleep(0.5)  # Be nice to arxiv API

        if metadata:
            lastname = get_author_lastname(metadata['author'])
            year = metadata['year']
            short_title = shorten_title(metadata['title'])

            new_name = f"{lastname}-{year}-{short_title}-{arxiv_id}.pdf"

            print(f"  Author: {metadata['author']}")
            print(f"  Title: {metadata['title'][:60]}...")
            print(f"  Year: {year}")
            print(f"  New name: {new_name}")
            print()

            renames.append((old_name, new_name, metadata))
        else:
            print(f"  Could not fetch metadata, skipping\n")

    # Save rename plan
    plan_file = PAPERS_DIR / "rename_plan.json"
    with open(plan_file, 'w') as f:
        json.dump([{'old': old, 'new': new, 'metadata': meta} for old, new, meta in renames], f, indent=2)
    print(f"\nSaved rename plan to {plan_file}")

    # Execute renames
    print("\nExecuting renames...")
    for old_name, new_name, _ in renames:
        old_path = PAPERS_DIR / old_name
        new_path = PAPERS_DIR / new_name
        if old_path.exists():
            os.rename(old_path, new_path)
            print(f"  {old_name} -> {new_name}")

    print(f"\nDone! Renamed {len(renames)} files.")

if __name__ == "__main__":
    main()
