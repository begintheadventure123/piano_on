import argparse
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE = "https://theremin.music.uiowa.edu"
INDEX_PAGE = f"{BASE}/MIS.html"
AUDIO_EXTS = (".wav", ".aiff", ".aif")


def is_non_piano_source(href: str) -> bool:
    h = href.lower()
    if not h.endswith(".html"):
        return False
    if "mis" not in h:
        return False
    if "piano" in h:
        return False
    return True


def discover_source_pages(session: requests.Session, include_2012: bool) -> list[str]:
    resp = session.get(INDEX_PAGE, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pages: set[str] = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if not is_non_piano_source(href):
            continue
        if (not include_2012) and ("2012" in href):
            continue
        pages.add(urljoin(f"{BASE}/", href))

    return sorted(pages)


def discover_audio_links(session: requests.Session, page_url: str) -> list[str]:
    resp = session.get(page_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links: set[str] = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        h = href.lower()
        if not h.endswith(AUDIO_EXTS):
            continue
        full = urljoin(f"{BASE}/", href)
        if "piano" in full.lower():
            continue
        links.add(full)
    return sorted(links)


def page_folder_name(page_url: str) -> str:
    stem = Path(page_url).stem
    return stem.replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Download public non-piano samples from University of Iowa")
    parser.add_argument("--out", default="ml/data/raw/non_piano/uiowa", help="Output folder")
    parser.add_argument("--limit", type=int, default=180, help="Max number of files to download")
    parser.add_argument("--limit-pages", type=int, default=18, help="Max number of source pages to crawl")
    parser.add_argument(
        "--include-2012",
        action="store_true",
        help="Include additional MIS-Pitches-2012 pages (larger and more varied set)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as s:
        source_pages = discover_source_pages(s, include_2012=args.include_2012)
        if not source_pages:
            raise RuntimeError("No non-piano source pages discovered from MIS index")

        source_pages = source_pages[: args.limit_pages]
        print(f"Discovered {len(source_pages)} source pages")

        all_items: list[tuple[str, str]] = []
        for page in source_pages:
            links = discover_audio_links(s, page)
            folder = page_folder_name(page)
            for url in links:
                all_items.append((folder, url))

        if not all_items:
            raise RuntimeError("No audio links discovered from selected source pages")

        selected = all_items[: args.limit]
        print(f"Discovered {len(all_items)} links, downloading {len(selected)}")

        for idx, (folder, url) in enumerate(selected, start=1):
            name = url.split("/")[-1]
            target_dir = out_dir / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name

            if target.exists() and target.stat().st_size > 0:
                print(f"[{idx}/{len(selected)}] exists {folder}/{name}")
                continue

            print(f"[{idx}/{len(selected)}] download {folder}/{name}")
            resp = s.get(url, timeout=60)
            resp.raise_for_status()
            target.write_bytes(resp.content)

    print(f"Done. Files in: {out_dir}")


if __name__ == "__main__":
    main()
