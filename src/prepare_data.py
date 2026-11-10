"""
Stap 1: data voorbereiden.

Wat doet dit script?
1. Leest het ruwe CBS-bestand (Kerncijfers wijken en buurten 2023, tabel 85618NED).
2. Houdt alleen buurten over (geen wijken, gemeenten of Nederland), zodat
   er geen dubbeltellingen ontstaan.
3. Controleert sleutels, gemeentecodes, percentages en ontbrekende waarden.
4. Geeft de kolommen begrijpelijke interne namen (zie config.json).
5. Schrijft:
   - data/processed/model_table.csv     (één rij per buurt)
   - data/processed/gemeente_table.csv  (door CBS gepubliceerde gemeentecijfers)
   - reports/source_manifest.json       (herkomst van de bron)
   - reports/data_quality.json          (controles en uitval)

Uitvoeren:  python src/prepare_data.py
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

# Alle paden zijn relatief ten opzichte van de projectmap.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pad(relatief_pad):
    """Maak een volledig pad binnen de projectmap."""
    return os.path.join(PROJECT_DIR, relatief_pad)


def lees_config():
    with open(pad("config.json"), encoding="utf-8") as f:
        return json.load(f)


def sha256_van_bestand(bestandspad):
    """Vingerafdruk van het ruwe bestand, zodat je kunt zien of het later is veranderd."""
    h = hashlib.sha256()
    with open(bestandspad, "rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blok)
    return h.hexdigest()


def lees_metadata(config):
    """Lees de officiële CBS-kolombeschrijvingen (DataProperties)."""
    with open(pad(config["metadata_file"]), encoding="utf-8") as f:
        eigenschappen = json.load(f)["value"]
    return {e["Key"]: e for e in eigenschappen if e.get("Key")}


def main():
    config = lees_config()
    os.makedirs(pad("data/processed"), exist_ok=True)
    os.makedirs(pad("reports"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Ruwe data inlezen. Codes lezen we als tekst (niet als getal).
    # ------------------------------------------------------------------
    ruw_pad = pad(config["raw_file"])
    ruw = pd.read_csv(ruw_pad, dtype={"Codering_3": str, "WijkenEnBuurten": str},
                      low_memory=False)

    # CBS-tekstvelden hebben vaak spaties aan het eind; die halen we weg.
    for kolom in ["SoortRegio_2", "Codering_3", "Gemeentenaam_1", "WijkenEnBuurten"]:
        ruw[kolom] = ruw[kolom].astype(str).str.strip()

    # Controle: bestaan alle kolommen uit config.json echt in het bestand?
    ontbrekende_kolommen = [k for k in config["column_mapping"] if k not in ruw.columns]
    if ontbrekende_kolommen:
        raise SystemExit(f"Deze kolommen ontbreken in het ruwe bestand: {ontbrekende_kolommen}")

    # ------------------------------------------------------------------
    # 2. Alleen buurten houden.
    # ------------------------------------------------------------------
    rijen_per_regiosoort = ruw["SoortRegio_2"].value_counts().to_dict()
    buurten = ruw[ruw["SoortRegio_2"] == "Buurt"].copy()

    # Sleutelcontrole: elke buurtcode mag maar één keer voorkomen.
    dubbele_buurtcodes = int(buurten["Codering_3"].duplicated().sum())
    if dubbele_buurtcodes > 0:
        raise SystemExit(f"{dubbele_buurtcodes} dubbele buurtcodes gevonden: eerst oplossen.")

    # Buurtcode = 'BU' + gemeentecode (4) + wijkcode (2) + buurtcode (2).
    # Dit staat zo in de CBS-metadata van kolom Codering_3.
    # Let op: de gemeentecode bestaat uit cijfers, maar wijk- en buurtcode kunnen
    # ook letters bevatten (bijvoorbeeld Amsterdam: BU0363AA01). Codes blijven dus tekst.
    codes_goed_formaat = buurten["Codering_3"].str.fullmatch(r"BU\d{4}[0-9A-Z]{4}")
    if not codes_goed_formaat.all():
        raise SystemExit("Niet alle buurtcodes hebben het verwachte formaat BU + 4 cijfers + 4 tekens.")
    codes_met_letters = int((~buurten["Codering_3"].str.fullmatch(r"BU\d{8}")).sum())
    buurten["gemeente_code"] = "GM" + buurten["Codering_3"].str[2:6]

    # Controle: hoort bij elke gemeentecode precies één gemeentenaam (en andersom)?
    namen_per_code = buurten.groupby("gemeente_code")["Gemeentenaam_1"].nunique().max()
    codes_per_naam = buurten.groupby("Gemeentenaam_1")["gemeente_code"].nunique().max()

    # Controle: bestaan de afgeleide gemeentecodes ook als gemeenterij in dezelfde tabel?
    gemeenten = ruw[ruw["SoortRegio_2"] == "Gemeente"].copy()
    codes_zonder_gemeenterij = sorted(set(buurten["gemeente_code"]) - set(gemeenten["Codering_3"]))

    # ------------------------------------------------------------------
    # 3. Interne namen geven en alleen de benodigde kolommen houden.
    # ------------------------------------------------------------------
    mapping = config["column_mapping"]
    modeltabel = buurten[list(mapping.keys()) + ["gemeente_code"]].rename(columns=mapping)
    modeltabel.insert(1, "target_year", config["target_year"])
    doel = config["target_internal_name"]

    # ------------------------------------------------------------------
    # 4. Kwaliteitscontroles.
    # ------------------------------------------------------------------
    # Percentages moeten tussen 0 en 100 liggen.
    percentage_kolommen = [k for k in modeltabel.columns if k.startswith("pct_")]
    buiten_bereik = {}
    for kolom in percentage_kolommen:
        waarden = modeltabel[kolom].dropna()
        buiten_bereik[kolom] = int(((waarden < 0) | (waarden > 100)).sum())

    # Complementaire paren: tellen samen op tot ongeveer 100% (afronding op hele procenten).
    def som_paar(a, b):
        s = (modeltabel[a] + modeltabel[b]).dropna()
        return {"min": float(s.min()), "max": float(s.max()), "aantal": int(len(s))}

    complementen = {
        "pct_eengezinswoning + pct_meergezinswoning": som_paar("pct_eengezinswoning", "pct_meergezinswoning"),
        "pct_bouwjaar_voor_2000 + pct_bouwjaar_vanaf_2000": som_paar("pct_bouwjaar_voor_2000", "pct_bouwjaar_vanaf_2000"),
    }

    # Negatieve waarden in gebruikte numerieke kolommen (zouden niet mogen).
    numeriek = modeltabel.select_dtypes("number")
    negatieve_waarden = {k: int(v) for k, v in (numeriek < 0).sum().items() if v > 0}

    # Stedelijkheid moet een code 1 t/m 5 zijn.
    stedelijkheid_codes = sorted(modeltabel["stedelijkheid"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # 5. Doelvariabele: ontbrekende waarden NIET invullen.
    # ------------------------------------------------------------------
    heeft_doel = modeltabel[doel].notna()
    modeltabel["has_label"] = heeft_doel
    modeltabel["uitval_reden"] = ""
    modeltabel.loc[~heeft_doel, "uitval_reden"] = (
        "Geen elektriciteitscijfer gepubliceerd (CBS vermeldt het alleen bij 6 of meer "
        "bewoonde woningen, of het cijfer is geheim/onbekend)"
    )

    # Wie valt er uit? Vergelijk buurten met en zonder doelwaarde.
    met_label = modeltabel[heeft_doel]
    zonder_label = modeltabel[~heeft_doel]
    uitval_vergelijking = {
        "mediaan_woningvoorraad_met_label": float(met_label["woningvoorraad"].median()),
        "mediaan_woningvoorraad_zonder_label": float(zonder_label["woningvoorraad"].median()),
        "aandeel_zonder_label_met_0_woningen": float((zonder_label["woningvoorraad"] == 0).mean()),
        "stedelijkheid_verdeling_met_label": met_label["stedelijkheid"].value_counts(normalize=True).sort_index().round(3).to_dict(),
        "stedelijkheid_verdeling_zonder_label": zonder_label["stedelijkheid"].value_counts(normalize=True).sort_index().round(3).to_dict(),
    }

    # Ontbrekende waarden per kenmerk, alleen binnen buurten mét doelwaarde.
    alle_kenmerken = sorted(set(sum(config["feature_sets"].values(), [])))
    missend_per_kenmerk = {k: int(met_label[k].isna().sum()) for k in alle_kenmerken}

    # ------------------------------------------------------------------
    # 6. Opslaan.
    # ------------------------------------------------------------------
    # Sleutel buurt_code + target_year moet uniek zijn.
    if modeltabel.duplicated(["buurt_code", "target_year"]).any():
        raise SystemExit("Sleutel buurt_code + target_year is niet uniek.")
    modeltabel.to_csv(pad("data/processed/model_table.csv"), index=False)

    # Door CBS zelf gepubliceerde gemeentecijfers (voor de gemeentevergelijking).
    gemeente_tabel = gemeenten[["Codering_3", "Gemeentenaam_1", "GemiddeldeElektriciteitsleveringTotaal_48",
                                "Woningvoorraad_35", "MateVanStedelijkheid_125"]].rename(columns={
        "Codering_3": "gemeente_code", "Gemeentenaam_1": "gemeente_naam",
        "GemiddeldeElektriciteitsleveringTotaal_48": "electricity_delivery_kwh_per_dwelling_year",
        "Woningvoorraad_35": "woningvoorraad", "MateVanStedelijkheid_125": "stedelijkheid"})
    gemeente_tabel.to_csv(pad("data/processed/gemeente_table.csv"), index=False)

    # Kenmerken van gemeenten en heel Nederland, met dezelfde kolomnamen als de buurten.
    # Alleen gebruikt om een buurt te vergelijken met haar gemeente en met Nederland (niet om te trainen).
    regio = ruw[ruw["SoortRegio_2"].isin(["Gemeente", "Land"])]
    regio_tabel = regio[list(mapping.keys())].rename(columns=mapping).rename(columns={"buurt_code": "regio_code",
                                                                                    "buurt_naam": "regio_naam"})
    regio_tabel.insert(0, "soort", regio["SoortRegio_2"].map({"Gemeente": "gemeente", "Land": "land"}).values)
    regio_tabel["regio_code"] = regio_tabel["regio_code"].str.strip()
    regio_tabel.to_csv(pad("data/processed/regio_kenmerken.csv"), index=False)

    # Herkomstregistratie (source manifest).
    metadata = lees_metadata(config)
    variabelen = []
    for ruwe_naam, interne_naam in mapping.items():
        m = metadata.get(ruwe_naam, {})
        variabelen.append({
            "ruwe_kolom": ruwe_naam,
            "interne_naam": interne_naam,
            "titel_cbs": m.get("Title"),
            "eenheid_cbs": m.get("Unit"),
            "beschrijving_cbs": " ".join((m.get("Description") or "").split()),
            "referentiejaar": config["target_year"],
            "ontbrekend_in_alle_buurten": int(buurten[ruwe_naam].isna().sum()),
        })
    tabel_info_pad = pad("data/metadata/85618NED_TableInfos.json")
    with open(tabel_info_pad, encoding="utf-8") as f:
        tabel_info = json.load(f)["value"][0]

    manifest = {
        "bronnen": [{
            "officiele_naam": config["source_title"],
            "tabelcode": config["source_table"],
            "url": config["source_url"],
            "statline_url": "https://www.cbs.nl/nl-nl/cijfers/detail/85618NED",
            "bestand": config["raw_file"],
            "sha256_ongewijzigd_bestand": sha256_van_bestand(ruw_pad),
            "bestand_gewijzigd_op": datetime.fromtimestamp(os.path.getmtime(ruw_pad)).isoformat(timespec="seconds"),
            "ophaalmethode": "CBS OData-API via het Python-pakket cbsodata (zie notebook Sprint3groep4final3.ipynb); "
                             "ophaaldatum niet vastgelegd, bestandsdatum hierboven is de best beschikbare indicatie",
            "bronversie_cbs_gewijzigd": tabel_info.get("Modified"),
            "periode": tabel_info.get("Period"),
            "status_cbs": "Definitief, tenzij bij het onderwerp anders vermeld (tabeltoelichting CBS)",
            "geografisch_niveau": "Gemeente, wijk en buurt; voor het model alleen buurt",
            "gebiedsindeling": "Wijk- en buurtindeling 2023",
            "extractiefilter": "Geen filter bij ophalen; filter SoortRegio_2 == 'Buurt' in prepare_data.py",
            "afronding_en_geheimhouding": "Energiecijfers afgerond op tientallen kWh en alleen vermeld bij 6 of meer "
                                         "(bewoonde) woningen; woningpercentages alleen bij 20 of meer woningen; "
                                         "aantallen op buurtniveau afgerond op vijftallen.",
            "metadata_bestanden": ["data/metadata/85618NED_DataProperties.json",
                                   "data/metadata/85618NED_TableInfos.json"],
            "metadata_opgehaald_op": datetime.fromtimestamp(os.path.getmtime(pad(config["metadata_file"]))).isoformat(timespec="seconds"),
            "variabelen": variabelen,
        }],
        "niet_gebruikte_bronnen": [
            {"tabelcode": "86159NED / 86333NED", "reden": "Niet aanwezig in de projectbestanden; zou een tweede jaar en een "
                                                           "aparte koppeling vereisen. Mogelijke uitbreiding."},
            {"tabelcode": "Enexis open data", "reden": "Postcodeniveau en gasdefinitie; hoort bij het aparte gastraject."},
        ],
    }
    with open(pad("reports/source_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    kwaliteit = {
        "rijen_ruw_bestand": int(len(ruw)),
        "rijen_per_regiosoort": rijen_per_regiosoort,
        "buurten": int(len(buurten)),
        "unieke_buurtcodes": int(buurten["Codering_3"].nunique()),
        "dubbele_buurtcodes": dubbele_buurtcodes,
        "buurtcodes_met_letters_in_wijk_of_buurtdeel": codes_met_letters,
        "gemeenten_in_buurten": int(buurten["gemeente_code"].nunique()),
        "max_namen_per_gemeentecode": int(namen_per_code),
        "max_codes_per_gemeentenaam": int(codes_per_naam),
        "gemeentecodes_zonder_gemeenterij": codes_zonder_gemeenterij,
        "buurten_met_doelwaarde": int(heeft_doel.sum()),
        "buurten_zonder_doelwaarde": int((~heeft_doel).sum()),
        "gemeenten_met_minstens_een_doelwaarde": int(met_label["gemeente_code"].nunique()),
        "doelwaarde_beschrijving_kwh": {k: float(v) for k, v in met_label[doel].describe().round(1).items()},
        "doelwaarde_geimputeerd": False,
        "missende_kenmerken_binnen_buurten_met_doel": missend_per_kenmerk,
        "percentages_buiten_0_100": buiten_bereik,
        "som_complementaire_percentages": complementen,
        "negatieve_waarden": negatieve_waarden,
        "stedelijkheid_codes": stedelijkheid_codes,
        "uitval_vergelijking": uitval_vergelijking,
        "joins": "Geen joins: doel en kenmerken komen uit dezelfde CBS-tabel en dezelfde gebiedsindeling.",
    }
    with open(pad("reports/data_quality.json"), "w", encoding="utf-8") as f:
        json.dump(kwaliteit, f, ensure_ascii=False, indent=2)

    print(f"Buurten: {len(buurten)} | met doelwaarde: {int(heeft_doel.sum())} | "
          f"gemeenten: {buurten['gemeente_code'].nunique()}")
    print("Geschreven: data/processed/model_table.csv, data/processed/gemeente_table.csv, "
          "reports/source_manifest.json, reports/data_quality.json")


if __name__ == "__main__":
    main()
