#!/usr/bin/env python3
"""
Script to verify that paper links in the course website match the actual papers.
Uses arxiv API to get metadata and compares with website descriptions.
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

PAPERS_DIR = Path(__file__).parent / 'papers'
SRC_DIR = Path(__file__).parent / 'src'


def get_arxiv_metadata(arxiv_id):
    """Fetch metadata from arxiv API."""
    # Clean up arxiv_id
    arxiv_id = arxiv_id.replace('arXiv:', '').strip()

    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        response = requests.get(api_url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)

        # Define namespace
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        title_elem = entry.find('atom:title', ns)
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else None

        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text.strip())

        summary_elem = entry.find('atom:summary', ns)
        summary = summary_elem.text.strip()[:500] if summary_elem is not None else None

        published_elem = entry.find('atom:published', ns)
        published = published_elem.text[:4] if published_elem is not None else None  # Just year

        return {
            'arxiv_id': arxiv_id,
            'title': title,
            'authors': authors,
            'year': published,
            'abstract': summary
        }

    except Exception as e:
        print(f"Error fetching metadata for {arxiv_id}: {e}")
        return None


def extract_paper_references_from_html(html_file):
    """Extract paper references and their context from HTML."""
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    references = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']

        if 'arxiv.org' not in href:
            continue

        # Extract arxiv ID
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)', href)
        if not match:
            continue

        arxiv_id = match.group(1)

        # Get surrounding context
        link_text = a_tag.get_text(strip=True)

        # Try to get the parent context
        parent = a_tag.parent
        if parent:
            # Get sibling text before the link
            context_parts = []
            for sibling in parent.children:
                if sibling == a_tag:
                    break
                if hasattr(sibling, 'get_text'):
                    context_parts.append(sibling.get_text(strip=True))
                elif isinstance(sibling, str):
                    context_parts.append(sibling.strip())

            parent_text = ' '.join(context_parts)

            # Also get full parent text
            full_parent = parent.get_text(strip=True)
        else:
            parent_text = ""
            full_parent = ""

        references.append({
            'arxiv_id': arxiv_id,
            'link_text': link_text,
            'context_before': parent_text[:200],
            'full_context': full_parent[:300],
            'source_file': html_file.name
        })

    return references


def compare_metadata(website_ref, arxiv_meta):
    """Compare website reference with arxiv metadata and report issues."""
    issues = []

    if arxiv_meta is None:
        return [f"Could not fetch metadata for arxiv:{website_ref['arxiv_id']}"]

    context = website_ref['full_context'].lower()
    title_words = arxiv_meta['title'].lower().split()

    # Check if title words appear in context
    title_matches = sum(1 for word in title_words[:5] if len(word) > 3 and word in context)

    # Check author names
    author_matches = 0
    first_author_last_name = arxiv_meta['authors'][0].split()[-1].lower() if arxiv_meta['authors'] else ""

    for author in arxiv_meta['authors'][:3]:  # Check first 3 authors
        last_name = author.split()[-1].lower()
        if last_name in context:
            author_matches += 1

    # Check year
    year_match = arxiv_meta['year'] and arxiv_meta['year'] in context

    # Determine confidence
    if author_matches == 0 and title_matches < 2:
        issues.append(f"LOW MATCH: Neither authors nor title found in context")
        issues.append(f"  Expected authors: {', '.join(arxiv_meta['authors'][:3])}")
        issues.append(f"  Expected title: {arxiv_meta['title'][:80]}...")
        issues.append(f"  Website context: {website_ref['full_context'][:150]}...")

    return issues


def main():
    print("=" * 70)
    print("PAPER VERIFICATION REPORT")
    print("=" * 70)

    # Load downloaded papers info
    results_file = PAPERS_DIR / 'download_results.json'
    with open(results_file, 'r') as f:
        download_results = json.load(f)

    downloaded = download_results['downloaded']

    # Get unique arxiv IDs from downloaded papers
    arxiv_ids = set()
    for item in downloaded:
        match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)', item['url'])
        if match:
            arxiv_ids.add(match.group(1))

    print(f"\nFound {len(arxiv_ids)} unique arxiv papers to verify\n")

    # Collect all references from HTML files
    html_files = list(SRC_DIR.glob('week*.html'))
    all_refs = []
    for html_file in sorted(html_files):
        refs = extract_paper_references_from_html(html_file)
        all_refs.extend(refs)

    # Group by arxiv_id
    refs_by_id = {}
    for ref in all_refs:
        aid = ref['arxiv_id']
        if aid not in refs_by_id:
            refs_by_id[aid] = []
        refs_by_id[aid].append(ref)

    # Verify each paper
    verification_results = []
    issues_found = []

    for i, arxiv_id in enumerate(sorted(arxiv_ids)):
        print(f"[{i+1}/{len(arxiv_ids)}] Verifying arxiv:{arxiv_id}...")

        # Get metadata from arxiv
        meta = get_arxiv_metadata(arxiv_id)
        time.sleep(0.5)  # Be nice to arxiv

        if meta is None:
            issues_found.append({
                'arxiv_id': arxiv_id,
                'issue': 'Could not fetch metadata from arxiv'
            })
            continue

        # Get website references
        refs = refs_by_id.get(arxiv_id, [])

        result = {
            'arxiv_id': arxiv_id,
            'title': meta['title'],
            'authors': meta['authors'],
            'year': meta['year'],
            'website_refs': refs,
            'issues': []
        }

        # Check each reference
        for ref in refs:
            issues = compare_metadata(ref, meta)
            if issues:
                result['issues'].extend(issues)
                result['issues'].append(f"  Source: {ref['source_file']}")

        verification_results.append(result)

        if result['issues']:
            issues_found.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal papers verified: {len(verification_results)}")
    print(f"Papers with potential issues: {len(issues_found)}")

    if issues_found:
        print("\n" + "-" * 70)
        print("PAPERS REQUIRING REVIEW:")
        print("-" * 70)

        for item in issues_found:
            print(f"\narxiv:{item['arxiv_id']}")
            if 'title' in item:
                print(f"  Title: {item['title'][:70]}...")
                print(f"  Authors: {', '.join(item['authors'][:3])}...")
            for issue in item.get('issues', []):
                print(f"  {issue}")

    # Save detailed results
    output_file = PAPERS_DIR / 'verification_report.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_verified': len(verification_results),
            'issues_count': len(issues_found),
            'all_results': verification_results,
            'issues': issues_found
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Print all papers with their actual metadata for manual review
    print("\n" + "=" * 70)
    print("ALL PAPERS - ACTUAL METADATA FROM ARXIV")
    print("=" * 70)

    for result in sorted(verification_results, key=lambda x: x['arxiv_id']):
        print(f"\narxiv:{result['arxiv_id']}")
        print(f"  Title: {result['title']}")
        print(f"  Authors: {', '.join(result['authors'][:4])}" + ("..." if len(result['authors']) > 4 else ""))
        print(f"  Year: {result['year']}")
        if result['website_refs']:
            for ref in result['website_refs']:
                print(f"  Website [{ref['source_file']}]: {ref['full_context'][:100]}...")


if __name__ == '__main__':
    main()
