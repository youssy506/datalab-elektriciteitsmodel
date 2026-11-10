"""
Stap 5: voorspellen met het opgeslagen model.

Deze module wordt gebruikt door de app (app.py) én voor batchverwerking.
Zo gaan alle voorspellingen via dezelfde route:
    invoer controleren  ->  opgeslagen pipeline  ->  voorspelling + waarschuwingen

Daarnaast (als src/explain.py is gedraaid):
    - een controle op ongewone COMBINATIES van kenmerken (artifacts/combinatiecheck.joblib)
    - verklaar(): per schatting de bijdrage van elk kenmerk (SHAP, TreeExplainer)

Batch gebruiken:
    python src/predict.py --input mijn_buurten.csv --output voorspellingen.csv
Het invoerbestand heeft één kolom per kenmerk (namen: zie artifacts/model_metadata.json).
"""

import argparse
import hashlib
import json
import math
import os

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Invoercontract: wat is een GELDIGE waarde per kenmerk (los van de trainingsdata)?
INVOER_SPEC = {
    "pct_meergezinswoning": {"label": "Aandeel meergezinswoningen (flats, appartementen)", "eenheid": "%",
                             "min": 0, "max": 100, "geheel_getal": False},
    "pct_bouwjaar_vanaf_2000": {"label": "Aandeel woningen gebouwd in 2000 of later", "eenheid": "%",
                                "min": 0, "max": 100, "geheel_getal": False},
    "gem_huishoudensgrootte": {"label": "Gemiddelde huishoudensgrootte", "eenheid": "personen per huishouden",
                               "min": 1, "max": 10, "geheel_getal": False},
    "stedelijkheid": {"label": "Stedelijkheid (CBS-klasse)", "eenheid": "1 = zeer sterk stedelijk ... 5 = niet stedelijk",
                      "min": 1, "max": 5, "geheel_getal": True},
    "woz_waarde_x1000": {"label": "Gemiddelde WOZ-waarde van woningen", "eenheid": "x 1.000 euro",
                         "min": 10, "max": 5000, "geheel_getal": False},
    "pct_koopwoningen": {"label": "Aandeel koopwoningen", "eenheid": "%",
                         "min": 0, "max": 100, "geheel_getal": False},
}


def pad(relatief_pad):
    return os.path.join(PROJECT_DIR, relatief_pad)


_CACHE = {}  # combinatiecheck en SHAP-uitlegger worden één keer geladen


def laad_model():
    """Laad de opgeslagen pipeline en de bijbehorende metadata."""
    pipeline = joblib.load(pad("artifacts/pipeline.joblib"))
    with open(pad("artifacts/model_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata


def pipeline_vingerafdruk():
    """SHA-256 van artifacts/pipeline.joblib. Uitleg- en controlebestanden bewaren deze, zodat ze
    nooit bij een ander (opnieuw getraind) model worden gebruikt, ook niet op dezelfde dag."""
    bestand = pad("artifacts/pipeline.joblib")
    sleutel = (bestand, os.path.getmtime(bestand))
    if _CACHE.get("hash_sleutel") != sleutel:
        with open(bestand, "rb") as f:
            _CACHE["hash"] = hashlib.sha256(f.read()).hexdigest()
        _CACHE["hash_sleutel"] = sleutel
    return _CACHE["hash"]


def nl_getal(waarde):
    """Getal in Nederlandse notatie: 388,5 en 1.234 (zonder overbodige nullen)."""
    tekst = f"{waarde:,.1f}".rstrip("0").rstrip(".")
    return tekst.replace(",", "X").replace(".", ",").replace("X", ".")


def is_leeg(waarde):
    return waarde is None or (isinstance(waarde, str) and waarde.strip() == "") or \
        (isinstance(waarde, float) and math.isnan(waarde))


def valideer(invoer, metadata, met_indicator=None):
    """Controleer één invoer (een dict met kenmerk -> waarde).

    Geeft terug: (schone_rij, fouten, waarschuwingen)
    - fouten: de invoer is ongeldig; er wordt GEEN voorspelling gemaakt.
    - waarschuwingen: voorspelling kan, maar is minder betrouwbaar.
    Een ontbrekende waarde mag (None of leeg), maar levert een waarschuwing op. Zijn ALLE kenmerken leeg,
    dan is dat een fout: er is dan niets om een schatting op te baseren.
    met_indicator: kenmerken waarvoor de pipeline een 'ontbrak'-kolom heeft (alleen voor de uitlegtekst).
    """
    kenmerken = metadata["kenmerken"]
    bereik = metadata["kenmerk_bereik_training"]
    fouten, waarschuwingen = [], []
    schoon = {}

    # Onverwachte of ontbrekende kenmerken
    onverwacht = [k for k in invoer if k not in kenmerken]
    if onverwacht:
        fouten.append(f"Onbekende kenmerken: {', '.join(onverwacht)}")
    afwezig = [k for k in kenmerken if k not in invoer]
    if afwezig:
        fouten.append(f"Deze kenmerken ontbreken in de invoer: {', '.join(afwezig)}")

    for k in kenmerken:
        if k not in invoer:
            continue
        spec = INVOER_SPEC[k]
        waarde = invoer[k]

        if is_leeg(waarde):
            schoon[k] = float("nan")
            tekst = (f"'{spec['label']}' is niet ingevuld. Het model vult intern de mediaan van de trainingsdata in "
                     f"({nl_getal(bereik[k]['mediaan'])} {spec['eenheid']})")
            if met_indicator is not None and k in met_indicator:
                tekst += (" én houdt bij dat de waarde ontbrak. De uitkomst is daarom niet gelijk aan het invullen "
                          "van de mediaan.")
            elif met_indicator is not None:
                tekst += (". Dit kenmerk ontbrak nooit in de trainingsdata, dus voor dit geval is het model niet "
                          "getest.")
            else:
                tekst += "."
            waarschuwingen.append(tekst + " In de test waren voorspellingen voor buurten met ontbrekende kenmerken "
                                          "duidelijk minder nauwkeurig.")
            continue

        # Datatype: moet een (eindig) getal zijn
        try:
            getal = float(str(waarde).replace(",", ".")) if isinstance(waarde, str) else float(waarde)
        except (TypeError, ValueError):
            fouten.append(f"'{spec['label']}' moet een getal zijn (ingevuld: {waarde!r}).")
            continue
        if not math.isfinite(getal):
            fouten.append(f"'{spec['label']}' moet een eindig getal zijn.")
            continue

        # Geldigheid (los van de trainingsdata)
        if getal < spec["min"] or getal > spec["max"]:
            fouten.append(f"'{spec['label']}' moet tussen {spec['min']} en {spec['max']} liggen "
                          f"({spec['eenheid']}); ingevuld: {nl_getal(getal)}.")
            continue
        if spec["geheel_getal"] and getal != int(getal):
            fouten.append(f"'{spec['label']}' moet een geheel getal zijn (1 t/m 5).")
            continue

        # Waarschuwing: buiten wat het model in de training heeft gezien
        b = bereik[k]
        if getal < b["min"] or getal > b["max"]:
            waarschuwingen.append(
                f"'{spec['label']}' = {nl_getal(getal)} valt buiten het bereik van de trainingsdata "
                f"({nl_getal(b['min'])} tot {nl_getal(b['max'])}). De voorspelling is hier niet getest.")
        elif getal < b["p01"] or getal > b["p99"]:
            waarschuwingen.append(
                f"'{spec['label']}' = {nl_getal(getal)} komt zelden voor in de trainingsdata "
                f"(98% ligt tussen {nl_getal(b['p01'])} en {nl_getal(b['p99'])}).")
        schoon[k] = getal

    if not fouten and schoon and all(isinstance(v, float) and math.isnan(v) for v in schoon.values()):
        fouten.append("Geen enkel kenmerk ingevuld: er is niets om een schatting op te baseren.")
    if fouten:
        return None, fouten, waarschuwingen
    return schoon, fouten, waarschuwingen


def indicator_kenmerken(pipeline, metadata):
    """Voor welke kenmerken heeft de pipeline een 'ontbrak'-kolom (alleen kenmerken die in de training ontbraken)?"""
    imputer = pipeline.named_steps["invullen"]
    if not hasattr(imputer, "indicator_") or imputer.indicator_ is None:
        return set()
    return {metadata["kenmerken"][j] for j in imputer.indicator_.features_}


def laad_combinatiecheck(metadata):
    """De detector voor ongewone combinaties (gemaakt door src/explain.py).

    Geeft None als het bestand ontbreekt of bij een ander model hoort; dan wordt de controle overgeslagen.
    """
    bestand = pad("artifacts/combinatiecheck.joblib")
    sleutel = (bestand, os.path.getmtime(bestand) if os.path.exists(bestand) else None)
    if _CACHE.get("combi_sleutel") != sleutel:
        _CACHE["combi_sleutel"] = sleutel
        _CACHE["combi"] = joblib.load(bestand) if sleutel[1] is not None else None
    combi = _CACHE["combi"]
    if (combi is None or combi.get("pipeline_sha256") != pipeline_vingerafdruk()
            or combi.get("kenmerken") != metadata["kenmerken"]):
        return None
    return combi


def ongewone_combinatie(rijen, pipeline, metadata):
    """True per rij als de combinatie van kenmerken ongewoner is dan ca. 99% van de trainingsbuurten.

    Alleen voor rijen waarin ALLE kenmerken bekend zijn (bij lege velden zou de ingevulde mediaan
    zelf een kunstmatige combinatie maken). Werkt op dezelfde geschaalde waarden als het model.
    """
    uit = np.zeros(len(rijen), dtype=bool)
    combi = laad_combinatiecheck(metadata)
    if combi is None or len(rijen) == 0:
        return uit
    kenmerken = metadata["kenmerken"]
    compleet = rijen[kenmerken].notna().all(axis=1).to_numpy()
    if compleet.any():
        geschaald = pipeline[:-1].transform(rijen.loc[compleet, kenmerken])[:, :len(kenmerken)]
        uit[compleet] = combi["detector"].predict(geschaald) == -1
    return uit


def combinatie_waarschuwing(metadata):
    """Tekst bij een ongewone combinatie, met wat de test daarover zegt."""
    v = laad_combinatiecheck(metadata)["validatie"]
    tekst = ("Deze combinatie van kenmerken is ongewoon: ze hoort bij de "
             f"{nl_getal(v['grens_pct'])}% meest afwijkende combinaties als je haar naast de trainingsbuurten legt. "
             "Elk kenmerk apart kan normaal zijn; het gaat om het samen voorkomen. ")
    return tekst + combinatie_testuitslag(v)


def combinatie_testuitslag(v):
    """Wat de test zegt over gemarkeerde buurten, met de regel 'pas vanaf 30 buurten een foutmaat'."""
    if v["test_n_gemarkeerd"] >= 30:
        return (f"In de test hadden zulke buurten een gemiddelde fout van {nl_getal(round(v['test_mae_gemarkeerd']))} "
                f"kWh, tegen {nl_getal(round(v['test_mae_overig']))} kWh bij de andere buurten.")
    tekst = (f"In de testgemeenten gebeurde dit maar bij {v['test_n_gemarkeerd']} van de {nl_getal(v['test_n_compleet'])} "
             "buurten met alle kenmerken bekend: te weinig om de fout betrouwbaar te meten")
    if v["test_mae_gemarkeerd"] is not None and v["test_mae_gemarkeerd"] > v["test_mae_overig"]:
        tekst += ", al was die in deze kleine groep groter dan bij de andere buurten"
    return tekst + ". Wees voorzichtig met deze schatting."


def voorspel(invoer, pipeline=None, metadata=None):
    """Voorspel voor één buurt. Geeft een dict met voorspelling, fouten en waarschuwingen."""
    if pipeline is None or metadata is None:
        pipeline, metadata = laad_model()
    schoon, fouten, waarschuwingen = valideer(invoer, metadata, indicator_kenmerken(pipeline, metadata))
    if fouten:
        return {"voorspelling_kwh": None, "fouten": fouten, "waarschuwingen": waarschuwingen}
    rij = pd.DataFrame([schoon], columns=metadata["kenmerken"])
    waarde = float(pipeline.predict(rij)[0])
    if ongewone_combinatie(rij, pipeline, metadata)[0]:
        waarschuwingen.append(combinatie_waarschuwing(metadata))
    return {"voorspelling_kwh": waarde, "fouten": [], "waarschuwingen": waarschuwingen}


def voorspel_batch(tabel, pipeline=None, metadata=None):
    """Voorspel voor een DataFrame met één rij per buurt (zelfde controles als de app).

    Elke rij wordt apart gecontroleerd; alle geldige rijen gaan daarna in één keer door het model
    (sneller dan rij voor rij, met dezelfde uitkomst).
    """
    if pipeline is None or metadata is None:
        pipeline, metadata = laad_model()
    kenmerken = metadata["kenmerken"]
    uitvoer_kolommen = ["voorspelling_kwh", "fouten", "waarschuwingen"]
    # Een eerder uitvoerbestand opnieuw inlezen mag: oude uitvoerkolommen worden vervangen, niet verdubbeld
    tabel = tabel.drop(columns=[c for c in uitvoer_kolommen if c in tabel.columns])
    met_indicator = indicator_kenmerken(pipeline, metadata)
    voorspellingen, fouten_tekst, waarschuwingen_lijst, geldig = [], [], [], {}
    for i, (_, rij) in enumerate(tabel.iterrows()):
        # Alleen modelkenmerken doorgeven; een ontbrekende kolom geeft een fout in valideer()
        invoer = {k: rij[k] for k in kenmerken if k in rij.index}
        schoon, fouten, waarschuwingen = valideer(invoer, metadata, met_indicator)
        voorspellingen.append(None)
        fouten_tekst.append(" | ".join(fouten))
        waarschuwingen_lijst.append(waarschuwingen)
        if not fouten:
            geldig[i] = schoon
    if geldig:
        rijen = pd.DataFrame(list(geldig.values()), columns=kenmerken)
        waarden = pipeline.predict(rijen)
        ongewoon = ongewone_combinatie(rijen, pipeline, metadata)
        for (i, _), waarde, vlag in zip(geldig.items(), waarden, ongewoon):
            voorspellingen[i] = float(waarde)
            if vlag:
                waarschuwingen_lijst[i].append(combinatie_waarschuwing(metadata))
    # Vaste kolommen, zodat ook een leeg invoerbestand een geldige (lege) uitvoer geeft
    uitvoer = pd.DataFrame({"voorspelling_kwh": pd.Series(voorspellingen, dtype="float64"),
                            "fouten": fouten_tekst,
                            "waarschuwingen": [" | ".join(w) for w in waarschuwingen_lijst]},
                           columns=uitvoer_kolommen)
    return pd.concat([tabel.reset_index(drop=True), uitvoer], axis=1)


def verklaar(invoer, pipeline=None, metadata=None):
    """Waarom deze schatting? Bijdrage van elk kenmerk in kWh (SHAP-waarden, TreeExplainer).

    Geeft {"basis": .., "bijdragen": {kenmerk: kWh}, "voorspelling_kwh": ..} of None bij ongeldige invoer.
    basis + som van de bijdragen = de schatting (op afronding na). Bij een leeg veld telt het effect
    van 'onbekend' (ingevulde mediaan + ontbreekt-indicator) bij dat kenmerk.
    """
    if pipeline is None or metadata is None:
        pipeline, metadata = laad_model()
    schoon, fouten, _ = valideer(invoer, metadata)
    if fouten:
        return None
    import shap  # pas laden als er iets uitgelegd moet worden (scheelt opstarttijd)

    model = pipeline.named_steps["model"]
    if _CACHE.get("uitlegger_model") is not model:
        _CACHE["uitlegger_model"] = model
        _CACHE["uitlegger"] = shap.TreeExplainer(model)
    uitlegger = _CACHE["uitlegger"]
    kenmerken = metadata["kenmerken"]
    rij = pd.DataFrame([schoon], columns=kenmerken)
    geschaald = pipeline[:-1].transform(rij)
    waarden = np.asarray(uitlegger.shap_values(geschaald)).reshape(-1)
    # Kolommen na het invullen: eerst de kenmerken, dan één 0/1-kolom per kenmerk dat in de training ontbrak
    indicator_van = [kenmerken[j] for j in pipeline.named_steps["invullen"].indicator_.features_]
    bijdragen = dict.fromkeys(kenmerken, 0.0)
    for naam, w in zip(list(kenmerken) + indicator_van, waarden):
        bijdragen[naam] += float(w)
    basis = float(np.ravel(uitlegger.expected_value)[0])
    return {"basis": basis, "bijdragen": bijdragen, "voorspelling_kwh": float(pipeline.predict(rij)[0])}


def main():
    parser = argparse.ArgumentParser(description="Batchvoorspelling elektriciteitslevering per woning")
    parser.add_argument("--input", required=True, help="CSV met één kolom per kenmerk")
    parser.add_argument("--output", required=True, help="CSV voor de voorspellingen")
    args = parser.parse_args()

    pipeline, metadata = laad_model()
    tabel = pd.read_csv(args.input)
    extra = [k for k in tabel.columns if k not in metadata["kenmerken"]]
    if extra:
        print(f"Let op: deze kolommen worden niet gebruikt door het model: {extra}")
    uit = voorspel_batch(tabel, pipeline, metadata)
    uit.to_csv(args.output, index=False)
    print(f"{uit['voorspelling_kwh'].notna().sum()} van {len(uit)} rijen voorspeld -> {args.output}")
    print(f"Eenheid: {metadata['doel']['eenheid']} (referentiejaar {metadata['doel']['referentiejaar']})")


if __name__ == "__main__":
    main()
