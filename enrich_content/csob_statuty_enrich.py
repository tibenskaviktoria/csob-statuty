"""
Obohatenie surových štatútov (statuty_raw.json) cez Claude API - v3.

Zmeny oproti v2 (dôvod: príprava na automatizáciu cez cron/GitHub Actions):

  1. AUTOMATICKÁ DETEKCIA ZMIEN namiesto ručného LIMIT/ONLY_TITLES.
     Každý raw záznam má teraz content_hash (SHA256 textu). Skript porovná
     hash oproti poslednému behu (uloženému v statuty_enriched.json) a
     enrichne len záznamy, ktoré sú nové alebo majú iný hash (napr. predĺžená
     platnosť, doplnená podmienka a pod.). LIMIT/ONLY_TITLES ostávajú ako
     voliteľné ručné prepnutie pre výnimočné prípady (napr. chceš prehnať
     všetko znova po zmene promptu), ale nie sú to už primárny mechanizmus.

  2. SELF-REFERENTIAL MERGE. Skript teraz merguje voči SVOJMU VLASTNÉMU
     predošlému výstupu (statuty_enriched.json), nie voči zamrznutému
     statuty_anthropic_enriched.json. To bolo potrebné opraviť - inak sa
     pri opakovanom behu vždy vraciame k starému snapshotu a strácame
     medzičasom pridané dáta.

  3. OPRAVENÉ PORADIE. Predošlá verzia stavala finálny zoznam ako
     list(merged.values()) - to pri prírastkovom behu posúva nové záznamy
     na koniec namiesto na ich skutočnú pozíciu na stránke. Teraz sa
     finálny zoznam vždy poskladá v poradí AKTUÁLNEHO raw scrapu.

  4. ZRUŠENÉ ŠTATÚTY. Ak štatút zmizne z aktuálneho scrapu (stiahnutý
     z webu), vypadne aj z aktívneho statuty_enriched.json a zaloguje sa
     do statuty_removed_log.json pre audit (nemaže sa ticho bez stopy).

Vyžaduje: pip install anthropic --break-system-packages
Očakáva premennú prostredia ANTHROPIC_API_KEY.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from anthropic import Anthropic

client = Anthropic()

MODEL = "claude-sonnet-5"
# Pre bežné doplnkové behy stačí spravidla "claude-haiku-4-5-20251001" -
# pri automatizovaných behoch, kde ide typicky o pár zmien denne, je rozdiel
# v cene zanedbateľný, tak necháme presnejší Sonnet.

# --- Voliteľné ručné prepnutie (zriedka potrebné) --------------------------
# LIMIT: ak nastavené, obmedzí AJ automaticky detegované zmeny na prvých N
# záznamov zo statuty_raw.json (None = bez obmedzenia).
LIMIT = None

# FORCE_TITLES: podreťazce názvov, ktoré sa MUSIA enrichnúť nech je hash
# akýkoľvek (napr. chceš prehnať konkrétny štatút znova po úprave promptu).
FORCE_TITLES: list[str] = []
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR.parent / "web_scraper" / "statuty_raw.json"
OUTPUT_PATH = BASE_DIR / "statuty_enriched.json"          # aj vstup aj výstup (self-referential)
REMOVED_LOG_PATH = BASE_DIR / "statuty_removed_log.json"

SYSTEM_PROMPT = """Si asistent pre pracovníkov infolinky slovenskej banky ČSOB.
Dostaneš text právneho štatútu súťaže/akcie. Tvojou úlohou je vytiahnuť z neho
prehľadné štruktúrované informácie cez nástroj extract_statut_info, ktoré
pomôžu agentovi na telefóne rýchlo pochopiť, o čo v akcii ide, bez toho, aby
musel čítať celý právnický text.

Pravidlá:
- Piš po slovensky, jasne, bez právnického žargónu.
- Ak dátum/suma nie je v texte jednoznačne uvedená, daj null - NEVYMÝŠĽAJ si.
- NIKDY nevynechaj konkrétnu sumu odmeny, ak je v texte uvedená.
- "short_title" NESMIE byť len skopírovaný marketingový názov, ak ten nič
  nehovorí o obsahu. Napíš krátky, vecný názov, ktorý hneď napovie, o čo ide.
- "categories" - vyber 1 až 2 z: "Účty a platby", "Úvery a lízing",
  "Investovanie", "Poistenie", "Hypotéky", "Karty", "Ostatné".
- "referenced_statutes" - presné citované názvy iných štatútov, na ktoré sa
  text odvoláva. Prázdny zoznam, ak žiadne nie sú.
- "search_keywords" napíš tak, ako by to povedal bežný klient telefonicky.
- Ak text spomína, že ide o aktualizáciu/dodatok/predĺženie predchádzajúcej
  verzie akcie, uveď to v "summary".
"""

TOOL_SCHEMA = {
    "name": "extract_statut_info",
    "description": "Uloží štruktúrované informácie vytiahnuté zo štatútu súťaže/akcie.",
    "input_schema": {
        "type": "object",
        "properties": {
            "short_title": {"type": "string"},
            "summary": {"type": "string"},
            "eligibility": {"type": ["string", "null"]},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "reward": {"type": "string"},
            "reward_amount_eur": {"type": ["number", "null"]},
            "date_from": {"type": ["string", "null"]},
            "date_to": {"type": ["string", "null"]},
            "payout_deadline": {"type": ["string", "null"]},
            "payout_note": {"type": "string"},
            "categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["Účty a platby", "Úvery a lízing", "Investovanie",
                             "Poistenie", "Hypotéky", "Karty", "Ostatné"],
                },
                "minItems": 1,
                "maxItems": 2,
            },
            "search_keywords": {"type": "array", "items": {"type": "string"}},
            "referenced_statutes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "short_title", "summary", "eligibility", "conditions", "reward",
            "reward_amount_eur", "date_from", "date_to", "payout_deadline",
            "payout_note", "categories", "search_keywords", "referenced_statutes",
        ],
    },
}


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def enrich_one(statute: dict, retries: int = 2) -> dict:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=[TOOL_SCHEMA],  # type: ignore[arg-type]
                tool_choice={"type": "tool", "name": "extract_statut_info"},
                messages=[
                    {
                    "role": "user",
                    "content": f"NÁZOV: {statute['title']}\n\nTEXT ŠTATÚTU:\n{statute['raw_text']}",
                    }
                ],
            )
            tool_block = next(b for b in resp.content if b.type == "tool_use")
            return tool_block.input
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise last_err  # type: ignore[misc]


def resolve_references(enriched: list[dict]) -> None:
    by_title = {e["title"]: e for e in enriched}
    titles = list(by_title.keys())

    def find_match(ref_name: str):
        ref_clean = ref_name.strip().strip("„“\"'").lower()
        for t in titles:
            if ref_clean in t.lower():
                return by_title[t]
        return None

    for e in enriched:
        resolved = []
        for ref in e.get("referenced_statutes", []) or []:
            match = find_match(ref)
            resolved.append({
                "name": ref,
                "url": match["url"] if match else None,
                "title": match["title"] if match else None,
            })
        e["referenced_statutes_resolved"] = resolved


def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        current_raw = json.load(f)

    for item in current_raw:
        item["content_hash"] = content_hash(item.get("raw_text", ""))

    previous_enriched = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            previous_enriched = json.load(f)
    previous_by_url = {e["url"]: e for e in previous_enriched}

    current_urls = {r["url"] for r in current_raw}
    removed = [e for url, e in previous_by_url.items() if url not in current_urls]

    to_process = []
    for r in current_raw:
        prev = previous_by_url.get(r["url"])
        is_new = prev is None
        is_changed = prev is not None and prev.get("content_hash") != r["content_hash"]
        is_forced = any(t in r["title"] for t in FORCE_TITLES)
        if is_new or is_changed or is_forced:
            to_process.append(r)

    if LIMIT:
        to_process = to_process[:LIMIT]

    print(f"Aktuálne na stránke: {len(current_raw)} | "
          f"na (pre)spracovanie: {len(to_process)} | "
          f"zrušené: {len(removed)}")

    to_process_urls = {r["url"] for r in to_process}
    newly_by_url = {}
    for i, statute in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] {statute['title'][:70]}...")
        try:
            data = enrich_one(statute)
            newly_by_url[statute["url"]] = {
                "title": statute["title"],
                "url": statute["url"],
                "source": statute.get("source", "aktualne"),
                "content_hash": statute["content_hash"],
                **data,
            }
        except Exception as e:
            print(f"  CHYBA aj po opakovaní: {e}")
            newly_by_url[statute["url"]] = {
                "title": statute["title"],
                "url": statute["url"],
                "source": statute.get("source", "aktualne"),
                "content_hash": statute["content_hash"],
                "short_title": statute["title"],
                "summary": "(automatické spracovanie zlyhalo, pozri originál)",
                "eligibility": None, "conditions": [], "reward": None,
                "reward_amount_eur": None, "date_from": None, "date_to": None,
                "payout_deadline": None, "payout_note": None,
                "categories": ["Ostatné"], "search_keywords": [],
                "referenced_statutes": [],
            }
        time.sleep(0.3)

    # Finálny zoznam VŽDY v poradí aktuálneho raw scrapu - toto garantuje
    # zhodu s poradím na oficiálnej stránke bez ohľadu na to, čo/kedy sa
    # prespracovalo.
    final_enriched = []
    for r in current_raw:
        if r["url"] in to_process_urls:
            final_enriched.append(newly_by_url[r["url"]])
        else:
            final_enriched.append(previous_by_url[r["url"]])

    resolve_references(final_enriched)

    if removed:
        removed_log = []
        if REMOVED_LOG_PATH.exists():
            removed_log = json.loads(REMOVED_LOG_PATH.read_text(encoding="utf-8"))
        for e in removed:
            removed_log.append({"url": e["url"], "title": e["title"]})
        REMOVED_LOG_PATH.write_text(
            json.dumps(removed_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ({len(removed)} zrušených záznamov zalogovaných do {REMOVED_LOG_PATH.name})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_enriched, f, ensure_ascii=False, indent=2)

    changes = len(to_process) + len(removed)
    print(f"\nHotovo. {OUTPUT_PATH.name}: {len(final_enriched)} záznamov.")
    print(f"ZMENY: {changes}")  # CI podľa tohto riadku rozhodne, či commitnúť


if __name__ == "__main__":
    main()
