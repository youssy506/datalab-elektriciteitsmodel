"""
Modelkaart (model card) volgens de opbouw van Mitchell et al. (2019), "Model Cards for Model Reporting".

Alle getallen komen uit de bestanden die de pipeline zelf schrijft (artifacts/model_metadata.json,
reports/metrics.json, reports/feature_importance.json, reports/combinatiecheck.json), zodat de kaart
niet uit de pas kan lopen met het model. Gebruikt door de app (tab "Uitleg") en door src/explain.py,
dat de kaart ook als MODEL_CARD.md wegschrijft.
"""

from predict import INVOER_SPEC, combinatie_testuitslag, nl_getal

KORT = {"pct_meergezinswoning": "meergezinswoningen", "pct_bouwjaar_vanaf_2000": "bouwjaar vanaf 2000",
        "gem_huishoudensgrootte": "huishoudensgrootte", "stedelijkheid": "stedelijkheid",
        "woz_waarde_x1000": "WOZ-waarde", "pct_koopwoningen": "koopwoningen"}
MAANDEN = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
           "november", "december"]


def _datum(iso):
    """'2026-11-10T..' -> '10 november 2026'."""
    jaar, maand, dag = iso[:10].split("-")
    return f"{int(dag)} {MAANDEN[int(maand) - 1]} {jaar}"

STEDELIJKHEID_NAMEN = {"1.0": "1 · zeer sterk", "2.0": "2 · sterk", "3.0": "3 · matig", "4.0": "4 · weinig",
                       "5.0": "5 · niet"}


def _zonder_niet(zin):
    """'Niet het verbruik van ...' -> 'Het verbruik van ...' (de kop zegt al 'níet')."""
    return zin[5:6].upper() + zin[6:] if zin.startswith("Niet ") else zin


def _kwh(waarde):
    return f"{nl_getal(round(float(waarde), -1))} kWh"


def secties(metadata, metrics, config, belang=None, combinatie=None, met_indicator=None):
    """Lijst van (titel, markdown) in de volgorde van de modelkaart.

    met_indicator: kenmerken met een 'ontbrak'-kolom in de pipeline (zie predict.indicator_kenmerken).
    """
    kandidaten = len([n for n in metrics["cross_validatie"] if n != "mediaanbaseline"])
    zonder_indicator = [k for k in metadata["kenmerken"] if met_indicator is not None and k not in met_indicator]
    doel = metadata["doel"]
    test = metadata["test"]
    rf = config["models"]["random_forest"]
    sw = metadata["software"]
    uit = []

    uit.append(("Modeldetails", "\n".join([
        f"- **Versie:** {metadata['model_versie']} (getraind op {_datum(metadata['getraind_op'])})",
        f"- **Type:** Random Forest-regressie ({rf['n_estimators']} bomen, maximale diepte {rf['max_depth']}, "
        f"minimale bladgrootte {rf['min_samples_leaf']}), gekozen uit {kandidaten} kandidaten (plus een "
        "mediaanbaseline als referentie) op de laagste MAE in 5-voudige cross-validatie.",
        "- **Voorbewerking:** een ontbrekend kenmerk krijgt de mediaan van de training. Voor kenmerken die in de "
        "training soms ontbraken komt er ook een 0/1-kolom 'ontbrak' bij"
        + (f" (niet voor {', '.join(KORT[k] for k in zonder_indicator)}, dat in de "
           "training nooit ontbrak)" if zonder_indicator else "")
        + ". Daarna standaardiseren. Alles zit in één opgeslagen scikit-learn-pipeline.",
        f"- **Software:** Python {sw['python']}, scikit-learn {sw['scikit-learn']}, pandas {sw['pandas']}, "
        f"numpy {sw['numpy']}. Een pipeline-bestand werkt alleen betrouwbaar met dezelfde scikit-learn-versie.",
        "- **Gemaakt door:** DataLab I, groep 4 (studentenproject).",
        "- **Licentie:** de CBS-gegevens vallen onder CC BY 4.0 (bron vermelden). Voor de code is geen "
        "licentie vastgelegd.",
        f"- **Citeren:** DataLab I, groep 4 ({metadata['getraind_op'][:4]}). *Voorspelmodel elektriciteitslevering "
        f"per woning*, versie {metadata['model_versie']}. Gegevens: {doel['bron']}.",
    ])))

    uit.append(("Wat de schatting meet", "\n".join(
        [doel["definitie"], "", f"- **Eenheid:** {doel['eenheid']}", f"- **Jaar:** {doel['referentiejaar']}",
         f"- **Niveau:** {doel['geografisch_niveau']}", "", "Wat de schatting níet is:"]
        + [f"- {_zonder_niet(zin)}" for zin in doel["wat_het_niet_is"]])))

    uit.append(("Bedoeld gebruik", "\n".join([
        f"- Een schatting van de **gemiddelde** jaarlijkse elektriciteitslevering uit het net per particuliere "
        f"woning in een Nederlandse buurt in {doel['referentiejaar']}, op basis van zes buurtkenmerken.",
        "- Verkennen hoe buurtkenmerken samenhangen met die levering, en buurten en gemeenten vergelijken.",
        "- Onderwijs: laten zien hoe je een voorspelmodel eerlijk test en uitlegt.",
    ])))

    uit.append(("Niet bedoeld voor", "\n".join([
        "- De energierekening of het verbruik van **één huishouden** (het is een buurtgemiddelde).",
        f"- Andere jaren dan {doel['referentiejaar']}, of buurten buiten Nederland.",
        "- Piekvermogen, netcongestie of netplanning.",
        "- Beslissingen over personen of huishoudens, of beleid zonder eigen controle van de cijfers.",
        "- Buurten met heel weinig woningen: daar publiceert het CBS meestal geen cijfer en is het model niet getest.",
    ])))

    kenmerk_regels = [f"- {INVOER_SPEC[k]['label']} – {INVOER_SPEC[k]['eenheid']}" for k in metadata["kenmerken"]]
    uit.append(("Gegevens", "\n".join([
        f"- **Bron:** {doel['bron']} (CBS-buurtcijfers {doel['referentiejaar']}).",
        f"- **Training:** {nl_getal(metadata['training']['n_buurten'])} buurten in "
        f"{metadata['training']['n_gemeenten']} gemeenten. **Test:** {nl_getal(test['n_buurten'])} buurten in "
        f"{test['n_gemeenten']} andere gemeenten (indeling per gemeente, zodat buurten uit dezelfde gemeente "
        "niet in training én test zitten).",
        "- Buurten zonder gepubliceerd CBS-cijfer doen niet mee aan training of test.",
        "- **Kenmerken:**",
    ] + ["  " + r for r in kenmerk_regels])))

    t, b = metrics["test_gekozen_model"], metrics["test_mediaanbaseline"]
    r2 = f"{t['R2']:.2f}".replace(".", ",")
    sub = metrics["fouten_per_subgroep_test"]
    groep_regels = [f"  - stedelijkheid {STEDELIJKHEID_NAMEN.get(k, k)}: {_kwh(v['MAE'])} ({nl_getal(v['n'])} buurten)"
                    for k, v in sub["stedelijkheid"].items() if "MAE" in v]
    groep_regels += [
        f"  - alle kenmerken bekend: {_kwh(sub['kenmerk_ontbreekt']['False']['MAE'])} "
        f"({nl_getal(sub['kenmerk_ontbreekt']['False']['n'])} buurten)",
        f"  - minstens één kenmerk ontbreekt: {_kwh(sub['kenmerk_ontbreekt']['True']['MAE'])} "
        f"({nl_getal(sub['kenmerk_ontbreekt']['True']['n'])} buurten)",
    ]
    uit.append(("Evaluatie (testgemeenten)", "\n".join([
        f"- **MAE** {_kwh(t['MAE'])} · **RMSE** {_kwh(t['RMSE'])} · **R²** {r2}",
        f"- Ter vergelijking, steeds de mediaan voorspellen: MAE {_kwh(b['MAE'])} "
        f"({nl_getal(metrics['test_verbetering_mae_tov_baseline_pct'])}% minder fout).",
        "- **MAE per groep:**",
    ] + groep_regels + [
        "- Het model trekt uitschieters naar het midden: buurten met een heel hoge levering worden te laag "
        "geschat, buurten met een heel lage levering te hoog (zie de tab Model).",
    ])))

    uitleg = ["- **Per schatting:** de app toont met SHAP-waarden hoeveel elk kenmerk de schatting omhoog of omlaag "
              "duwt ten opzichte van de gemiddelde schatting (SHAP: Lundberg & Lee, 2017; berekend met TreeExplainer: "
              "Lundberg et al., 2020). Bij een leeg veld is dat het effect van 'onbekend'.",
              "- Dit verklaart het **model**, niet de werkelijkheid: het zijn samenhangen, geen oorzaken."]
    if belang:
        top = sorted(belang["kenmerken"].items(), key=lambda kv: -kv[1]["toename_mae_gemiddeld"])
        uitleg.insert(1, "- **Belang over alle testbuurten** (hoeveel stijgt de MAE als je een kenmerk door elkaar "
                         "husselt): " + "; ".join(f"{INVOER_SPEC[k]['label']} +{nl_getal(round(v['toename_mae_gemiddeld']))} kWh"
                                                  for k, v in top) + ". Meergezinswoningen, koopwoningen, "
                         "huishoudensgrootte en stedelijkheid hangen sterk samen (rangcorrelatie 0,6 à 0,7 in de "
                         "training) en kunnen elkaar deels vervangen; hun belang wordt zo te laag ingeschat.")
    uit.append(("Uitleg van het model", "\n".join(uitleg)))

    beperkingen = [
        "- **Zonnepanelen** ontbreken als kenmerk, terwijl ze de levering uit het net direct verlagen.",
        "- **Samenhang, geen oorzaak.** WOZ-waarde en koopwoningen zijn indirecte aanwijzingen voor woninggrootte "
        "en huishoudtype; ze hangen ook onderling samen.",
        "- **Ontbrekende kenmerken** geven een veel grotere fout (zie evaluatie).",
        "- **Afronding.** CBS rondt energiecijfers af op tientallen kWh.",
        "- **Geen betrouwbaarheidsinterval per schatting.** Dat is niet onderzocht en wordt daarom niet getoond; "
        "de test-MAE is een gemiddelde, geen garantie.",
    ]
    if combinatie:
        v = combinatie
        beperkingen.append(
            f"- **Ongewone combinaties** worden gemarkeerd als alle kenmerken bekend zijn (Local Outlier Factor, "
            f"grens bij de {nl_getal(v['grens_pct'])}% meest afwijkende trainingsbuurten). "
            + combinatie_testuitslag(v).replace(" Wees voorzichtig met deze schatting.", ""))
    uit.append(("Beperkingen en risico's", "\n".join(beperkingen)))
    return uit


def markdown(metadata, metrics, config, belang=None, combinatie=None, met_indicator=None):
    tekst = [f"# Modelkaart: elektriciteitslevering per woning (CBS-buurt, {metadata['doel']['referentiejaar']})", ""]
    for titel, inhoud in secties(metadata, metrics, config, belang, combinatie, met_indicator):
        tekst += [f"## {titel}", "", inhoud, ""]
    tekst.append("_Opbouw volgens Mitchell et al. (2019), Model Cards for Model Reporting. "
                 "Gegenereerd door src/explain.py uit de modelbestanden._")
    return "\n".join(tekst) + "\n"
