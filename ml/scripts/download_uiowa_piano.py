import argparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://theremin.music.uiowa.edu"
PAGE = f"{BASE}/MISpiano.html"


def discover_links(session: requests.Session):
    r = session.get(PAGE, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href_l = href.lower()
        if not (href_l.endswith(".aiff") or href_l.endswith(".wav")):
            continue
        if href.startswith("http"):
            url = href
        else:
            url = f"{BASE}/{href.lstrip('/')}"
        links.append(url)
    return sorted(set(links))


def main():
    parser = argparse.ArgumentParser(description="Download public piano note samples from University of Iowa")
    parser.add_argument("--out", default="ml/data/raw/piano/uiowa", help="Output folder")
    parser.add_argument("--limit", type=int, default=60, help="Max number of files to download")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as s:
        urls = discover_links(s)
        if not urls:
            raise RuntimeError("No sample links discovered from source page")

        selected = urls[: args.limit]
        print(f"Discovered {len(urls)} links, downloading {len(selected)}")

        for idx, url in enumerate(selected, start=1):
            name = url.split("/")[-1]
            target = out_dir / name
            if target.exists() and target.stat().st_size > 0:
                print(f"[{idx}/{len(selected)}] exists {name}")
                continue

            print(f"[{idx}/{len(selected)}] download {name}")
            resp = s.get(url, timeout=60)
            resp.raise_for_status()
            target.write_bytes(resp.content)

    print(f"Done. Files in: {out_dir}")


if __name__ == "__main__":
    main()
