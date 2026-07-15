"""
Obohatenie surových štatútov (statuty_raw.json) cez Claude API - v2.

Zmeny oproti v1:
  - Používa Anthropic "tool use" namiesto voľného textového JSON výstupu.
    Model je donútený vyplniť presne definovanú schému cez tool_choice,
    takže odpadá krehké parsovanie textu a chyby typu "model pridal text
    navyše" alebo "nedodržaná štruktúra" (čo sme videli pri lokálnom modeli
    aj pri 2 zlyhaniach v predošlom behu - jeden bol spôsobený tým, že názov
    štatútu obsahoval rovnú úvodzovku ", ktorá pri voľnom texte rozbila JSON).
  - "categories" je teraz zoznam (1-2 položky) - niektoré štatúty sa týkajú
    viacerých oblastí naraz (napr. poistenie + účet).
  - "short_title" má prísnejšiu inštrukciu: musí byť vypovedajúci o obsahu
    aj vtedy, keď je marketingový názov nič nehovoriaci (napr. "Odporúčate
    byť smart? IV", "Perspektiv Plus 5 - Svetový výber").
  - "referenced_statutes": zoznam presných citovaných názvov iných štatútov,
    na ktoré sa text odvoláva (napr. "Pozvaní sa riadia štatútom
    „Každý nákup je investíciou“") - po behu sa tieto názvy priradia
    k reálnym URL v post-processing kroku nižšie.
  - Voliteľné spracovanie len podmnožiny (LIMIT / ONLY_TITLES) a zlúčenie
    výsledkov s už existujúcim enriched súborom podľa URL, aby sa dal
    projekt spracovávať postupne bez straty predošlej práce.

Vyžaduje: pip install anthropic --break-system-packages
Očakáva premennú prostredia ANTHROPIC_API_KEY.
"""

import json
import os
import time
from pathlib import Path

from anthropic import Anthropic

client = Anthropic()

MODEL = "claude-sonnet-5"  # pre túto dávku (zložitejšie/referenčné štatúty) Sonnet
# Pre bežné doplnkové behy stačí spravidla "claude-haiku-4-5-20251001"

# --- Konfigurácia rozsahu spracovania -------------------------------------
# LIMIT: spracuje len prvých N záznamov zo statuty_raw.json (None = všetky).
# Stránka radí štatúty od najnovších, takže LIMIT=40 = 40 najaktuálnejších.
LIMIT = 40

# ONLY_TITLES: podreťazce názvov, ktoré sa MUSIA spracovať bez ohľadu na LIMIT
# (napr. predtým zlyhané záznamy).
ONLY_TITLES = [
    "Moja garáž",
    "Preleťte do ČSOB 2025",
]

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR.parent / "web_scraper" / "statuty_raw.json"
EXISTING_ENRICHED_PATH = BASE_DIR / "statuty_anthropic_enriched.json"  # ak existuje, zlúči sa
OUTPUT_PATH = BASE_DIR / "statuty_enriched.json"
# ---------------------------------------------------------------------------

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
  nehovorí o obsahu (napr. "Odporúčate byť smart? IV", "Perspektiv Plus 5 -
  Svetový výber"). Namiesto toho napíš krátky, vecný názov, ktorý hneď
  napovie, o čo ide (napr. "Odmena za odporučenie nového klienta",
  "Fond Perspektiv Plus 5 - globálne akcie"). Ak marketingový názov už
  sám osebe je vypovedajúci, môžeš ho ponechať/mierne skrátiť.
- "categories" - vyber 1 až 2 z: "Účty a platby", "Úvery a lízing",
  "Investovanie", "Poistenie", "Hypotéky", "Karty", "Ostatné". Ak sa akcia
  týka viacerých oblastí súčasne (napr. podmienkou je poistenie AJ účet),
  priraď obe.
- "referenced_statutes" - ak text spomína, že sa účastník/podmienky riadia
  INÝM štatútom (napr. "riadi sa štatútom „Každý nákup je investíciou“"),
  ulož sem presný citovaný názov toho druhého štatútu presne tak, ako je
  napísaný v texte (s úvodzovkami vnútri reťazca je to v poriadku, tool use
  to spracuje správne). Ak žiadny taký odkaz nie je, vráť prázdny zoznam.
- "search_keywords" napíš tak, ako by to povedal bežný klient telefonicky.
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


def enrich_one(statute: dict, retries: int = 2) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
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
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to enrich statute after retries")


def resolve_references(enriched: list[dict]) -> None:
    """Skús priradiť referenced_statutes k reálnym URL v rámci datasetu."""
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
        all_statutes = json.load(f)

    to_process = list(all_statutes[:LIMIT]) if LIMIT else list(all_statutes)
    already_titles = {s["title"] for s in to_process}
    for s in all_statutes:
        if s["title"] not in already_titles and any(t in s["title"] for t in ONLY_TITLES):
            to_process.append(s)

    print(f"Spracujem {len(to_process)} z {len(all_statutes)} záznamov.")

    newly_enriched = []
    for i, statute in enumerate(to_process, 1):
        print(f"[{i}/{len(to_process)}] {statute['title'][:70]}...")
        try:
            data = enrich_one(statute)
            newly_enriched.append({
                "title": statute["title"],
                "url": statute["url"],
                "source": statute.get("source", "aktualne"),
                **data,
            })
        except Exception as e:
            print(f"  CHYBA aj po opakovaní: {e}")
            newly_enriched.append({
                "title": statute["title"],
                "url": statute["url"],
                "source": statute.get("source", "aktualne"),
                "short_title": statute["title"],
                "summary": "(automatické spracovanie zlyhalo, pozri originál)",
                "eligibility": None, "conditions": [], "reward": None,
                "reward_amount_eur": None, "date_from": None, "date_to": None,
                "payout_deadline": None, "payout_note": None,
                "categories": ["Ostatné"], "search_keywords": [],
                "referenced_statutes": [],
            })
        time.sleep(0.3)

    # Zlúčenie s existujúcim súborom (podľa URL), ak existuje.
    merged = {}
    if os.path.exists(EXISTING_ENRICHED_PATH):
        with open(EXISTING_ENRICHED_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for e in existing:
            # zabezpeč konzistentnú štruktúru so starými záznamami (category -> categories)
            if "categories" not in e and "category" in e:
                e["categories"] = [e.pop("category")]
            e.setdefault("referenced_statutes", [])
            merged[e["url"]] = e

    for e in newly_enriched:
        merged[e["url"]] = e  # nové/prepracované záznamy prepíšu staré

    final = list(merged.values())
    resolve_references(final)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"\nHotovo. {OUTPUT_PATH}: {len(final)} záznamov spolu "
          f"({len(newly_enriched)} nových/prepracovaných).")


if __name__ == "__main__":
    main()
