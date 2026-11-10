"""
Tests voor de app zelf (zonder browser), met Streamlit AppTest.
Controleert de inhoudsstatussen uit reports/design/design_contract.md.

Uitvoeren: .venv/bin/python -m pytest -v
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(PROJECT_DIR, "app.py")


def start():
    return AppTest.from_file(APP, default_timeout=60).run()


def bereken(at):
    knop = next(b for b in at.button if b.label == "Schatting berekenen")
    return knop.click().run()


def teksten(at):
    """Alle zichtbare tekst uit markdown, metrics, meldingen en captions."""
    delen = [m.value for m in at.markdown] + [c.value for c in at.caption]
    delen += [f"{m.label} {m.value}" for m in at.metric]
    delen += [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(str(d) for d in delen)


def test_laadt_zonder_fouten_en_toont_lege_status():
    at = start()
    assert not at.exception
    assert "Nog geen schatting" in teksten(at)
    assert len(at.tabs) == 5


def test_voorbeeld_vult_velden_en_geeft_schatting():
    at = start()
    at.selectbox(key="voorbeeld").select("Typische matig stedelijk buurt").run()
    assert at.session_state["in_stedelijkheid"] == 3
    at = bereken(at)
    assert not at.exception
    tekst = teksten(at)
    assert "Geschatte levering uit het net" in tekst and "kWh" in tekst
    assert "Hoger dan" in tekst
    assert "Gedeeltelijke invoer" not in tekst
    assert not at.error


def test_lege_velden_geven_gedeeltelijke_status():
    at = start()
    at.number_input(key="in_pct_meergezinswoning").set_value(20).run()
    at = bereken(at)
    tekst = teksten(at)
    assert "Geschatte levering uit het net" in tekst
    assert "Gedeeltelijke invoer" in tekst


def test_afwijkende_invoer_geeft_waarschuwing():
    at = start()
    at.selectbox(key="voorbeeld").select("Typische matig stedelijk buurt").run()
    at.number_input(key="in_woz_waarde_x1000").set_value(3000).run()
    at = bereken(at)
    assert any("buiten het bereik" in w.value for w in at.warning)


def test_ongeldige_invoer_geeft_foutstatus():
    # De invoervelden laten geen waarden buiten 0-100 toe; we testen de foutroute via session state,
    # zoals een verkeerd gevulde sessie of toekomstige invoerbron dat zou doen.
    at = start()
    at.session_state["resultaat"] = None
    at.selectbox(key="voorbeeld").select("Typische matig stedelijk buurt").run()
    from predict import laad_model, voorspel
    pipeline, metadata = laad_model()
    invoer = {k: at.session_state[f"in_{k}"] for k in metadata["kenmerken"]}
    invoer["pct_koopwoningen"] = 150
    at.session_state["resultaat"] = {"invoer": invoer, **voorspel(invoer, pipeline, metadata)}
    at.run()
    assert "Kan geen schatting maken" in teksten(at)
    assert at.error


def test_bestaande_buurt_toont_status_en_waarden():
    at = start()
    tekst = teksten(at)
    assert "CBS-waarde 2023" in tekst and "Schatting model" in tekst


def test_gemeententab_toont_rang():
    at = start()
    assert "Plek (laag → hoog)" in teksten(at)


def test_helemaal_leeg_verzenden_geeft_foutstatus():
    at = bereken(start())
    assert "Kan geen schatting maken" in teksten(at)
    assert any("minstens één kenmerk" in e.value for e in at.error)


def test_voorbeeld_wisselen_haalt_oud_resultaat_weg():
    at = start()
    at.selectbox(key="voorbeeld").select("Typische matig stedelijk buurt").run()
    at = bereken(at)
    assert "Geschatte levering uit het net" in teksten(at)
    at.selectbox(key="voorbeeld").select("Alles leegmaken").run()
    assert "Nog geen schatting" in teksten(at)


def kies_buurt(at, code):
    import pandas as pd
    tabel = pd.read_csv(os.path.join(PROJECT_DIR, "data/processed/model_table.csv"), dtype={"buurt_code": str})
    gemeente = tabel.set_index("buurt_code").loc[code, "gemeente_naam"]
    at.selectbox(key="buurt_gemeente").select(gemeente).run()
    return at.selectbox(key="buurt_code").select(code).run()


def badges(at):
    return [m.value for m in at.markdown if "-badge[" in m.value]


def test_buurt_zonder_woningen_krijgt_geen_schatting():
    at = kies_buurt(start(), "BU07722310")  # Poeijers, Eindhoven: 0 woningen, geen CBS-waarde
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["CBS-waarde 2023"] == "niet gepubliceerd"
    assert metrics["Schatting model"] == "geen woningen"
    assert any("Niet gebruikt" in b for b in badges(at))


def test_testbuurt_toont_groene_status():
    import pandas as pd
    split = pd.read_csv(os.path.join(PROJECT_DIR, "reports/split_membership.csv"), dtype={"buurt_code": str})
    code = split[split["set"] == "test"]["buurt_code"].iloc[0]
    at = kies_buurt(start(), code)
    assert not at.exception
    assert any("In de testset" in b for b in badges(at))


def test_buurt_met_ontbrekende_kenmerken_geeft_een_waarschuwing():
    import pandas as pd
    tabel = pd.read_csv(os.path.join(PROJECT_DIR, "data/processed/model_table.csv"), dtype={"buurt_code": str})
    code = tabel[tabel["has_label"] & tabel["pct_meergezinswoning"].isna()]["buurt_code"].iloc[0]
    at = kies_buurt(start(), code)
    gedeeltelijk = [w for w in at.warning if "Gedeeltelijke gegevens" in w.value]
    assert len(gedeeltelijk) == 1
    assert not any("niet ingevuld" in w.value for w in at.warning)


def test_gelijke_gemeentecijfers_delen_een_plek():
    at = start()  # Eindhoven heeft dezelfde afgeronde waarde als een andere gemeente
    plek = {m.label: m.value for m in at.metric}["Plek (laag → hoog)"]
    assert plek.startswith("gedeeld ")


def test_ontbrekend_model_geeft_foutmelding(tmp_path):
    import shutil
    kopie = tmp_path / "project"
    shutil.copytree(PROJECT_DIR, kopie, ignore=shutil.ignore_patterns(".venv", "artifacts", "report", ".pytest_cache"))
    # Apart proces: anders hergebruikt Python de al geladen predict-module en de Streamlit-cache
    import subprocess
    import sys
    script = ("from streamlit.testing.v1 import AppTest\n"
              "at = AppTest.from_file('app.py', default_timeout=60).run()\n"
              "print('EXC', len(at.exception)); print('ERR', at.error[0].value if at.error else '')\n")
    uit = subprocess.run([sys.executable, "-c", script], cwd=kopie, capture_output=True, text=True, timeout=120)
    assert "EXC 0" in uit.stdout, uit.stdout + uit.stderr
    assert "konden niet worden geladen" in uit.stdout, uit.stdout + uit.stderr


@pytest.fixture(autouse=True)
def _pad():
    import sys
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))


# ----------------------------------------------------------------------------------------------
# Uitbreidingen: postcode, eigen verbruik, uitleg, vergelijken, downloads, modelkaart
# ----------------------------------------------------------------------------------------------
def postcode_zoeken(at, tekst):
    at.text_input(key="postcode").input(tekst)
    knop = next(b for b in at.button if b.label == "Zoeken")
    return knop.click().run()


def test_postcode_ongeldig_geeft_melding_zonder_internet():
    at = postcode_zoeken(start(), "12AB")
    assert not at.exception
    assert any("geen geldige postcode" in e.value for e in at.error)


def test_postcode_zet_gemeente_en_buurt(monkeypatch):
    import postcode
    gevonden = [{"buurt_code": "BU0363AE02", "buurt_naam": "Oude Kerk e.o.", "gemeente_naam": "Amsterdam", "adressen": 27},
                {"buurt_code": "BU0363AE04", "buurt_naam": "Nes e.o.", "gemeente_naam": "Amsterdam", "adressen": 6}]
    monkeypatch.setattr(postcode, "zoek_buurten", lambda pc, timeout=6: ("1012JS", gevonden))
    at = postcode_zoeken(start(), "1012 js")
    assert not at.exception
    assert at.session_state["buurt_gemeente"] == "Amsterdam"
    assert at.session_state["buurt_code"] == "BU0363AE02"
    assert any("Oude Kerk" in s.value and "Nes e.o." in s.value for s in at.success)


def test_postcode_met_onbekende_buurtcode(monkeypatch):
    import postcode
    gevonden = [{"buurt_code": "BU07729999", "buurt_naam": "Nieuwbouw", "gemeente_naam": "Eindhoven", "adressen": 3}]
    monkeypatch.setattr(postcode, "zoek_buurten", lambda pc, timeout=6: ("5611AA", gevonden))
    at = postcode_zoeken(start(), "5611AA")
    assert not at.exception
    assert any("bestaat niet in de CBS-indeling" in w.value for w in at.warning)


def test_eigen_verbruik_in_buurttab():
    at = start()
    at.number_input(key="eigen_kwh_buurt").set_value(4000).run()
    assert not at.exception
    assert "Jouw levering" in teksten(at) and "één huishouden" in teksten(at)


def test_schatting_heeft_uitleg_eigen_verbruik_en_download():
    at = start()
    at.selectbox(key="voorbeeld").select("Typische matig stedelijk buurt").run()
    at = bereken(at)
    at.number_input(key="eigen_kwh_schatting").set_value(2000).run()
    assert not at.exception
    assert any(e.label == "Waarom deze schatting?" for e in at.expander)
    assert "lager** dan deze schatting" in teksten(at)
    knoppen = [d.proto.label for d in at.get("download_button")]
    assert "Schatting downloaden (CSV)" in knoppen


def test_vergelijking_met_gemeente_en_nederland():
    at = start()
    tabellen = [d.value.columns.tolist() for d in at.dataframe]
    assert any("Nederland" in k and "Eindhoven (gemeente)" in k for k in tabellen)


def test_uitleg_tabel_telt_op():
    """De tabel onder 'Waarom deze schatting?' begint bij het gemiddelde en eindigt bij de schatting."""
    at = start()
    uitleg = next(d.value for d in at.dataframe if d.value.columns.tolist() == ["Stap", "Waarde", "Bijdrage"])
    assert uitleg["Stap"].iloc[0] == "Gemiddelde schatting" and uitleg["Stap"].iloc[-1] == "Deze schatting"
    schatting = {m.label: m.value for m in at.metric}["Schatting model"]
    assert uitleg["Waarde"].iloc[-1] == schatting


def test_downloads_en_modelkaart_aanwezig():
    at = start()
    knoppen = {d.proto.label for d in at.get("download_button")}
    assert {"Ranglijst downloaden (CSV)", "Schattingen voor alle buurten (CSV)",
            "Modelkaart downloaden (.md)", "Voorbeeldbestand downloaden"} <= knoppen
    tekst = teksten(at)
    assert "Niet bedoeld voor" in tekst and "Op welke kenmerken leunt het model het meest?" in tekst


def test_postcodemelding_verdwijnt_bij_andere_buurt(monkeypatch):
    import postcode
    gevonden = [{"buurt_code": "BU0363AE02", "buurt_naam": "Oude Kerk e.o.", "gemeente_naam": "Amsterdam", "adressen": 27}]
    monkeypatch.setattr(postcode, "zoek_buurten", lambda pc, timeout=6: ("1012JS", gevonden))
    at = postcode_zoeken(start(), "1012JS")
    assert any("Oude Kerk" in s.value for s in at.success)
    at = at.selectbox(key="buurt_code").select("BU0363AE04").run()  # zelf een andere buurt kiezen
    assert not any("Oude Kerk" in s.value for s in at.success)


def test_vergelijking_twee_buurten_met_dezelfde_naam():
    import pandas as pd
    tabel = pd.read_csv(os.path.join(PROJECT_DIR, "data/processed/model_table.csv"), dtype={"buurt_code": str})
    dubbel = tabel[tabel.duplicated("buurt_naam", keep=False)].sort_values(["buurt_naam", "buurt_code"])
    a, b = dubbel.iloc[0], dubbel.iloc[1]
    assert a["buurt_naam"] == b["buurt_naam"]
    at = kies_buurt(start(), a["buurt_code"])
    at.selectbox(key="vergelijk_gemeente").select(b["gemeente_naam"]).run()
    at = at.selectbox(key="vergelijk_buurt").select(b["buurt_code"]).run()
    assert not at.exception
    kolommen = next(d.value.columns.tolist() for d in at.dataframe if "Nederland" in d.value.columns)
    assert any(a["buurt_code"] in k for k in kolommen) and any(b["buurt_code"] in k for k in kolommen)
