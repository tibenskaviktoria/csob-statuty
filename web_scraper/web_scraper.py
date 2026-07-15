"""
Scraper pre "Štatúty súťaží a akcií" na csob.sk

Čo robí:
1. Stiahne zoznam aktuálnych štatútov (aj archív) so správnymi odkazmi.
2. Pre každý štatút stiahne detail a vytiahne celý text.
3. Uloží všetko do jedného JSON súboru (surové dáta - vstup pre ďalší
   krok, ktorý by mal text obohatiť cez LLM - viď csob_statuty_enrich.py).

Vyžaduje: pip install requests beautifulsoup4 --break-system-packages

Poznámka: stránka je server-rendered, takže netreba Selenium/Playwright.
Ak by sa štruktúra HTML v budúcnosti zmenila, treba upraviť selektory
nižšie (sú okomentované, kde presne čo hľadajú).
"""

import json
import re
import time
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.csob.sk"
LISTING_URLS = [
    "https://www.csob.sk/dolezite-dokumenty/statuty-sutazi",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (interny nastroj infolinky CSOB; kontakt: vtibenska@csob.sk)"
}

# Zdvorilá pauza medzi requestami, aby sme stránku nezaťažovali.
DELAY_SECONDS = 1.0


@dataclass
class Statute:
    title: str
    url: str
    source: str  # "aktualne" or "archiv"
    raw_text: str | None = None


def download_listing(listing_url: str) -> list[Statute]:
    """Download one listing page and extract statute links."""
    resp = requests.get(listing_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    source = "archiv" if "archiv" in listing_url else "aktualne"
    statutes = []

    # Links to individual statutes use /-/name?redirect=... or full URLs with /-/.
    # The link text often starts with "Štatút" or related labels.
    for a in soup.find_all("a", href=True):
        href_value = a["href"]
        if not isinstance(href_value, str):
            continue

        href = href_value
        text = a.get_text(strip=True)
        if not text:
            continue
        if re.search(r"/-/[^/]+\?redirect=", href) or (
            href.startswith(BASE_URL) and "/-/" in href
        ):
            if text.lower().startswith(("štatút", "podmienky", "súťažný poriadok", "všeobecné pravidlá")):
                full_url = urljoin(BASE_URL, href)
                statutes.append(Statute(title=text, url=full_url, source=source))

    # Deduplicate by URL base without query string.
    unique_statutes = {}
    for statute in statutes:
        key = statute.url.split("?")[0]
        if key not in unique_statutes:
            unique_statutes[key] = statute
    return list(unique_statutes.values())


def download_detail(statute: Statute) -> str:
    """Download statute detail page and extract plain text."""
    resp = requests.get(statute.url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    articles = soup.find_all("div", class_=re.compile(r"\bjournal-content-article\b"))

    container = None
    for article in articles:
        heading = article.find("h1")
        h1_text = heading.get_text(strip=True) if heading else ""
        if h1_text and h1_text != "Štatúty súťaží a akcií":
            container = article
            break
    
    if container is None and len(articles) > 1:
        container = articles[1]
    
    if container is None and articles:
        container = articles[0]
    
    if not container:
        return ""

    for row in container.find_all("div", class_="row"):
        if row.find("h1") is not None:
            continue
        row.decompose()

    for tag in container.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    text = container.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main():
    statutes: list[Statute] = []

    print("Downloading listings...")
    for listing_url in LISTING_URLS:
        listing = download_listing(listing_url)
        print(f"  {listing_url} -> {len(listing)} links")
        statutes.extend(listing)
        time.sleep(DELAY_SECONDS)

    print(f"\nFound {len(statutes)} unique current statutes. Downloading details...")
    for i, statute in enumerate(statutes, 1):
        try:
            statute.raw_text = download_detail(statute)
            print(f"  [{i}/{len(statutes)}] OK: {statute.title}")
        except Exception as e:
            print(f"  [{i}/{len(statutes)}] ERROR fetching {statute.url}: {e}")
        time.sleep(DELAY_SECONDS)

    output = [asdict(s) for s in statutes]
    with open("statuty_raw.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved statuty_raw.json ({len(output)} records).")


if __name__ == "__main__":
    main()