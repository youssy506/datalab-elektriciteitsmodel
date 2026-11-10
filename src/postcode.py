"""
Postcode -> CBS-buurt, via de PDOK Locatieserver (gratis overheidsdienst, geen sleutel nodig).

We vragen adressen op met die postcode en tellen in welke buurt(en) ze liggen. Eén postcode kan in
meer dan één buurt vallen; dan geven we alle buurten terug, de buurt met de meeste adressen eerst.

Let op: PDOK gebruikt de HUIDIGE buurtindeling. Onze gegevens zijn van 2023. Een buurtcode die sindsdien
is veranderd, staat dan niet in onze tabel; de app meldt dat en laat de gemeente zien.

Documentatie: https://api.pdok.nl/bzk/locatieserver/search/v3_1/ui/
"""

import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

PDOK_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
POSTCODE_PATROON = re.compile(r"^\s*([1-9][0-9]{3})\s*([A-Za-z]{2})\s*$")


class PostcodeFout(Exception):
    """Fout met een tekst die de gebruiker kan lezen."""


def normaliseer(postcode):
    """'1012 js' -> '1012JS'. Geeft None als het geen geldige Nederlandse postcode is."""
    m = POSTCODE_PATROON.match(postcode or "")
    return f"{m.group(1)}{m.group(2).upper()}" if m else None


def lees_documenten(antwoord):
    """De adressen uit een PDOK-antwoord, plus het totale aantal. Tegen onverwachte vormen bestand."""
    response = antwoord.get("response") if isinstance(antwoord, dict) else None
    if not isinstance(response, dict):
        return [], 0
    docs = response.get("docs")
    docs = [d for d in docs if isinstance(d, dict)] if isinstance(docs, list) else []
    totaal = response.get("numFound")
    return docs, totaal if isinstance(totaal, int) else len(docs)


def lees_antwoord(antwoord):
    """Zet een PDOK-antwoord (dict) om in een lijst buurten, de buurt met de meeste adressen eerst."""
    docs, _ = lees_documenten(antwoord)
    return tel_buurten(docs)


def tel_buurten(docs):
    teller = Counter((d.get("buurtcode"), d.get("buurtnaam"), d.get("gemeentenaam"))
                     for d in docs if isinstance(d.get("buurtcode"), str) and d.get("buurtcode"))
    return [{"buurt_code": code, "buurt_naam": naam, "gemeente_naam": gemeente, "adressen": n}
            for (code, naam, gemeente), n in teller.most_common()]


PER_PAGINA = 100      # meer geeft PDOK niet per keer
MAX_PAGINAS = 10      # ruim genoeg: een postcode heeft zelden meer dan een paar honderd adressen


def zoek_buurten(postcode, timeout=6):
    """Vraag PDOK welke buurt(en) bij een postcode horen. Gooit PostcodeFout bij problemen."""
    schoon = normaliseer(postcode)
    if schoon is None:
        raise PostcodeFout("Dat is geen geldige postcode. Gebruik vier cijfers en twee letters, zoals 5611 AA.")
    alle_docs, start, totaal = [], 0, None
    try:
        while start < MAX_PAGINAS * PER_PAGINA and (totaal is None or start < totaal):
            vraag = urllib.parse.urlencode({"q": f"postcode:{schoon}", "fq": "type:adres", "rows": PER_PAGINA,
                                            "start": start, "fl": "buurtcode,buurtnaam,gemeentenaam"})
            with urllib.request.urlopen(f"{PDOK_URL}?{vraag}", timeout=timeout) as reactie:
                docs, totaal = lees_documenten(json.load(reactie))
            alle_docs += docs
            if not docs:
                break
            start += PER_PAGINA
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as fout:
        raise PostcodeFout("De postcodedienst van PDOK is nu niet bereikbaar (geen internet?). "
                           "Kies de gemeente en buurt hieronder zelf.") from fout
    buurten = tel_buurten(alle_docs)
    if not buurten:
        raise PostcodeFout(f"Geen adressen gevonden voor postcode {schoon[:4]} {schoon[4:]}.")
    return schoon, buurten
