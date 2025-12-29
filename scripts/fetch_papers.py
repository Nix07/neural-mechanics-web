#!/usr/bin/env python3
"""
Script to fetch papers linked in the course website and save them to a papers/ directory.
Handles arxiv, ACL Anthology, PNAS, and other sources with appropriate user agents.
"""

import os
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
import json

# User agent that mimics a browser (helps with arxiv and other sites)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

# Directory setup
SRC_DIR = Path(__file__).parent / 'src'
PAPERS_DIR = Path(__file__).parent / 'papers'
PAPERS_DIR.mkdir(exist_ok=True)

def extract_links_from_html(html_file):
    """Extract all paper links from an HTML file."""
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        text = a_tag.get_text(strip=True)
        parent_text = ""

        # Get surrounding context (for paper descriptions)
        parent = a_tag.parent
        if parent:
            parent_text = parent.get_text(strip=True)[:200]

        # Filter for paper links
        if any(domain in href for domain in [
            'arxiv.org',
            'aclanthology.org',
            'pnas.org',
            'transformer-circuits.pub',
            'distill.pub',
            'openreview.net',
            'proceedings.neurips.cc',
            'proceedings.mlr.cc',
            'github.com',  # for repos with papers
        ]):
            links.append({
                'url': href,
                'text': text,
                'context': parent_text,
                'source_file': html_file.name
            })

    return links

def get_arxiv_pdf_url(url):
    """Convert an arxiv abstract URL to a PDF URL."""
    # Handle various arxiv URL formats
    # https://arxiv.org/abs/2301.05217 -> https://arxiv.org/pdf/2301.05217.pdf
    # https://arxiv.org/abs/2301.05217v1 -> https://arxiv.org/pdf/2301.05217v1.pdf

    if '/abs/' in url:
        pdf_url = url.replace('/abs/', '/pdf/')
        if not pdf_url.endswith('.pdf'):
            pdf_url += '.pdf'
        return pdf_url
    elif '/pdf/' in url:
        if not url.endswith('.pdf'):
            return url + '.pdf'
        return url
    return None

def get_safe_filename(url, link_text):
    """Generate a safe filename from URL and link text."""
    # Try to extract arxiv ID
    arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)', url)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        # Clean up link text for filename
        clean_text = re.sub(r'[^\w\s-]', '', link_text)[:50].strip()
        clean_text = re.sub(r'\s+', '_', clean_text)
        return f"arxiv_{arxiv_id}_{clean_text}.pdf"

    # For other URLs, use the last path component
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    if path_parts:
        base = path_parts[-1]
        if not base.endswith('.pdf'):
            base += '.pdf'
        return base

    return None

def download_paper(url, link_info, session):
    """Download a paper from the given URL."""
    try:
        # Handle arxiv specially
        if 'arxiv.org' in url:
            pdf_url = get_arxiv_pdf_url(url)
            if not pdf_url:
                print(f"  Could not determine PDF URL for: {url}")
                return None
        else:
            pdf_url = url

        filename = get_safe_filename(url, link_info['text'])
        if not filename:
            print(f"  Could not determine filename for: {url}")
            return None

        filepath = PAPERS_DIR / filename

        # Skip if already downloaded
        if filepath.exists():
            print(f"  Already exists: {filename}")
            return filepath

        print(f"  Downloading: {pdf_url}")
        print(f"  -> {filename}")

        response = session.get(pdf_url, headers=HEADERS, timeout=30, allow_redirects=True)
        response.raise_for_status()

        # Check if we got a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' in content_type or response.content[:4] == b'%PDF':
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  Success: {filepath.name}")
            return filepath
        else:
            print(f"  Warning: Response doesn't appear to be a PDF (content-type: {content_type})")
            # Save anyway for inspection
            with open(filepath.with_suffix('.html'), 'wb') as f:
                f.write(response.content)
            return None

    except requests.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        return None

def fetch_metadata_from_arxiv(arxiv_id, session):
    """Fetch metadata from arxiv API."""
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        response = session.get(api_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'xml')
        entry = soup.find('entry')
        if entry:
            title = entry.find('title')
            authors = entry.find_all('author')
            summary = entry.find('summary')

            return {
                'title': title.get_text(strip=True) if title else None,
                'authors': [a.find('name').get_text(strip=True) for a in authors if a.find('name')],
                'abstract': summary.get_text(strip=True) if summary else None,
            }
    except Exception as e:
        print(f"  Error fetching metadata for {arxiv_id}: {e}")
    return None

def main():
    # Find all HTML files in src/
    html_files = list(SRC_DIR.glob('week*.html'))
    html_files.extend(SRC_DIR.glob('index.html'))

    print(f"Found {len(html_files)} HTML files")

    # Extract all links
    all_links = []
    for html_file in sorted(html_files):
        print(f"\nProcessing: {html_file.name}")
        links = extract_links_from_html(html_file)
        all_links.extend(links)
        print(f"  Found {len(links)} paper links")

    # Deduplicate by URL
    seen_urls = set()
    unique_links = []
    for link in all_links:
        if link['url'] not in seen_urls:
            seen_urls.add(link['url'])
            unique_links.append(link)

    print(f"\n{'='*60}")
    print(f"Total unique paper links: {len(unique_links)}")
    print(f"{'='*60}")

    # Save link inventory
    inventory_file = PAPERS_DIR / 'inventory.json'
    with open(inventory_file, 'w') as f:
        json.dump(unique_links, f, indent=2)
    print(f"\nSaved link inventory to: {inventory_file}")

    # Download papers (arxiv only for now)
    session = requests.Session()
    downloaded = []
    failed = []
    skipped = []

    print(f"\n{'='*60}")
    print("Downloading papers...")
    print(f"{'='*60}")

    for i, link in enumerate(unique_links, 1):
        url = link['url']
        print(f"\n[{i}/{len(unique_links)}] {link['text'][:60]}...")
        print(f"  URL: {url}")
        print(f"  Source: {link['source_file']}")

        # Only download arxiv papers for now (they have consistent PDF access)
        if 'arxiv.org' in url:
            result = download_paper(url, link, session)
            if result:
                downloaded.append({'url': url, 'file': str(result), **link})
            else:
                failed.append(link)

            # Be nice to arxiv
            time.sleep(1)
        else:
            print(f"  Skipping non-arxiv link")
            skipped.append(link)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped (non-arxiv): {len(skipped)}")

    # Save results
    results_file = PAPERS_DIR / 'download_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'downloaded': downloaded,
            'failed': failed,
            'skipped': skipped
        }, f, indent=2)
    print(f"\nSaved results to: {results_file}")

    # Print downloaded files
    print(f"\n{'='*60}")
    print("Downloaded papers:")
    print(f"{'='*60}")
    for item in downloaded:
        print(f"  - {Path(item['file']).name}")

    # Print failed downloads
    if failed:
        print(f"\n{'='*60}")
        print("Failed downloads:")
        print(f"{'='*60}")
        for item in failed:
            print(f"  - {item['url']}")

if __name__ == '__main__':
    main()
