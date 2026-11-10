"""
Tests voor de uitbreidingen: uitleg per schatting (SHAP), controle op ongewone combinaties,
kenmerkbelang, modelkaart, postcode-zoeker, vergelijkingsgegevens en CSV-uploads.

Uitvoeren (na src/explain.py): .venv/bin/python -m pytest -v
"""

import io
import json
import os
import sys
import urllib.error

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import model_card  # noqa: E402
import postcode  # noqa: E402
from predict import indicator_kenmerken, laad_model, verklaar, voorspel, voorspel_batch  # noqa: E402


def pad(p):
    return os.path.join(PROJECT_DIR, p)


def lees_json(p):
    with open(pad(p), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model():
    return laad_model()


@pytest.fixture(scope="module")
def modeltabel():
    return pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})


TYPISCH = {"pct_meergezinswoning": 20, "pct_bouwjaar_vanaf_2000": 10, "gem_huishoudensgrootte": 2.3,
           "stedelijkheid": 3, "woz_waarde_x1000": 340, "pct_koopwoningen": 65}
# Elk kenmerk apart komt voor, maar samen niet: bijna alleen flats, grote huishoudens, platteland, dure koopwoningen
ONGEWOON = {"pct_meergezinswoning": 95, "pct_bouwjaar_vanaf_2000": 90, "gem_huishoudensgrootte": 3.6,
            "stedelijkheid": 5, "woz_waarde_x1000": 1200, "pct_koopwoningen": 100}


# --- Uitleg per schatting (SHAP) ---------------------------------------------------------------
@pytest.mark.parametrize("invoer", [TYPISCH, dict(TYPISCH, woz_waarde_x1000=None, pct_koopwoningen=None), ONGEWOON])
def test_uitleg_telt_op_tot_de_schatting(model, invoer):
    uitleg = verklaar(invoer, *model)
    schatting = voorspel(invoer, *model)["voorspelling_kwh"]
    assert set(uitleg["bijdragen"]) == set(model[1]["kenmerken"])
    assert uitleg["basis"] + sum(uitleg["bijdragen"].values()) == pytest.approx(schatting, abs=1e-6)


def test_uitleg_bij_ongeldige_invoer_is_leeg(model):
    assert verklaar(dict(TYPISCH, pct_koopwoningen=150), *model) is None


# --- Ongewone combinaties ----------------------------------------------------------------------
def test_ongewone_combinatie_geeft_waarschuwing(model):
    tekst = " ".join(voorspel(ONGEWOON, *model)["waarschuwingen"])
    assert "combinatie van kenmerken is ongewoon" in tekst


def test_gewone_en_gedeeltelijke_invoer_geen_combinatiewaarschuwing(model):
    for invoer in [TYPISCH, dict(ONGEWOON, gem_huishoudensgrootte=None)]:
        assert not any("combinatie" in w for w in voorspel(invoer, *model)["waarschuwingen"])


def test_combinatiecheck_hoort_bij_dit_model_en_klopt_met_test(model):
    validatie = lees_json("reports/combinatiecheck.json")
    assert validatie["model_versie"] == model[1]["model_versie"]
    # ongeveer 1% van de training is gemarkeerd (zo is de grens gekozen)
    assert 0.5 <= validatie["aandeel_training_gemarkeerd_pct"] <= 1.5
    assert validatie["test_n_gemarkeerd"] <= validatie["test_n_compleet"]


# --- Batch: één aanroep geeft hetzelfde als rij voor rij ----------------------------------------
def test_batch_gelijk_aan_rij_voor_rij(model, modeltabel):
    kenmerken = model[1]["kenmerken"]
    rijen = pd.concat([modeltabel[kenmerken].sample(60, random_state=1), pd.DataFrame([ONGEWOON])],
                      ignore_index=True)
    batch = voorspel_batch(rijen, *model)
    for i, rij in rijen.iterrows():
        los = voorspel({k: rij[k] for k in kenmerken}, *model)
        if los["voorspelling_kwh"] is None:
            assert pd.isna(batch.loc[i, "voorspelling_kwh"])
        else:
            assert batch.loc[i, "voorspelling_kwh"] == pytest.approx(los["voorspelling_kwh"], abs=1e-9)
        assert batch.loc[i, "waarschuwingen"] == " | ".join(los["waarschuwingen"])


def test_csv_met_puntkomma_en_decimale_komma(model):
    """Zo schrijft de app downloads en zo leest de upload ze terug (Excel-NL-formaat)."""
    tabel = pd.DataFrame([dict(TYPISCH, naam="a"), dict(TYPISCH, naam="b", gem_huishoudensgrootte=None)])
    bestand = tabel.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    assert b"2,3" in bestand
    terug = pd.read_csv(io.BytesIO(bestand), sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    uit = voorspel_batch(terug, *model)
    assert uit["voorspelling_kwh"].notna().all()
    assert uit.loc[0, "voorspelling_kwh"] == pytest.approx(voorspel(TYPISCH, *model)["voorspelling_kwh"])
    assert "naam" in uit.columns


# --- Kenmerkbelang en modelkaart ---------------------------------------------------------------
def test_kenmerkbelang_compleet(model):
    belang = lees_json("reports/feature_importance.json")
    assert belang["model_versie"] == model[1]["model_versie"]
    assert set(belang["kenmerken"]) == set(model[1]["kenmerken"])
    assert all(v["toename_mae_sd"] >= 0 for v in belang["kenmerken"].values())


def test_modelkaart_bestand_loopt_gelijk_met_model(model):
    with open(pad("MODEL_CARD.md"), encoding="utf-8") as f:
        bestand = f.read()
    verwacht = model_card.markdown(model[1], lees_json("reports/metrics.json"), lees_json("config.json"),
                                   lees_json("reports/feature_importance.json"),
                                   lees_json("reports/combinatiecheck.json"), indicator_kenmerken(*model))
    assert bestand == verwacht
    for kop in ["Modeldetails", "Bedoeld gebruik", "Niet bedoeld voor", "Gegevens", "Evaluatie", "Beperkingen"]:
        assert f"## {kop}" in bestand


# --- Vergelijkingsgegevens ---------------------------------------------------------------------
def test_regio_kenmerken_gemeenten_en_nederland():
    regio = pd.read_csv(pad("data/processed/regio_kenmerken.csv"))
    gemeenten = pd.read_csv(pad("data/processed/gemeente_table.csv"))
    assert (regio["soort"] == "land").sum() == 1
    assert (regio["soort"] == "gemeente").sum() == len(gemeenten)
    samen = regio[regio["soort"] == "gemeente"].merge(gemeenten, left_on="regio_code", right_on="gemeente_code")
    assert len(samen) == len(gemeenten)
    kolom = "electricity_delivery_kwh_per_dwelling_year"
    assert np.allclose(samen[f"{kolom}_x"], samen[f"{kolom}_y"], equal_nan=True)


# --- Postcode ---------------------------------------------------------------------------------
@pytest.mark.parametrize("ruw, schoon", [("5611 aa", "5611AA"), (" 1012JS ", "1012JS"), ("0123AB", None),
                                         ("12345", None), ("", None)])
def test_postcode_normaliseren(ruw, schoon):
    assert postcode.normaliseer(ruw) == schoon


def test_postcode_antwoord_meeste_adressen_eerst():
    docs = ([{"buurtcode": "BU0363AE04", "buurtnaam": "Nes e.o.", "gemeentenaam": "Amsterdam"}] * 6
            + [{"buurtcode": "BU0363AE02", "buurtnaam": "Oude Kerk e.o.", "gemeentenaam": "Amsterdam"}] * 27)
    buurten = postcode.lees_antwoord({"response": {"docs": docs}})
    assert [b["buurt_code"] for b in buurten] == ["BU0363AE02", "BU0363AE04"]
    assert buurten[0]["adressen"] == 27


def test_postcode_ongeldig_of_geen_internet(monkeypatch):
    with pytest.raises(postcode.PostcodeFout, match="geen geldige postcode"):
        postcode.zoek_buurten("abc")

    def geen_netwerk(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(postcode.urllib.request, "urlopen", geen_netwerk)
    with pytest.raises(postcode.PostcodeFout, match="niet bereikbaar"):
        postcode.zoek_buurten("5611AA")


def test_postcode_echt_opzoeken(modeltabel):
    """Echte aanvraag bij PDOK; wordt overgeslagen zonder internet."""
    try:
        schoon, buurten = postcode.zoek_buurten("5611AA")
    except postcode.PostcodeFout as fout:
        pytest.skip(f"PDOK niet bereikbaar: {fout}")
    assert schoon == "5611AA"
    assert buurten[0]["buurt_code"] in set(modeltabel["buurt_code"])


# --- Regressietests uit de review ---------------------------------------------------------------
def test_eigen_uitvoer_opnieuw_inlezen_geeft_geen_dubbele_kolommen(model):
    eerste = voorspel_batch(pd.DataFrame([TYPISCH]), *model)
    tweede = voorspel_batch(eerste, *model)
    assert list(tweede.columns).count("voorspelling_kwh") == 1
    assert tweede.loc[0, "voorspelling_kwh"] == pytest.approx(eerste.loc[0, "voorspelling_kwh"])


def test_helemaal_lege_rij_krijgt_geen_schatting(model):
    leeg = {k: None for k in TYPISCH}
    assert voorspel(leeg, *model)["voorspelling_kwh"] is None
    uit = voorspel_batch(pd.DataFrame([TYPISCH, leeg]), *model)
    assert pd.isna(uit.loc[1, "voorspelling_kwh"]) and "Geen enkel kenmerk" in uit.loc[1, "fouten"]


def test_lege_stedelijkheid_zegt_niet_dat_er_een_markering_is(model):
    tekst = " ".join(voorspel(dict(TYPISCH, stedelijkheid=None), *model)["waarschuwingen"])
    assert "nooit in de trainingsdata" in tekst and "houdt bij" not in tekst
    tekst = " ".join(voorspel(dict(TYPISCH, pct_koopwoningen=None), *model)["waarschuwingen"])
    assert "houdt bij dat de waarde ontbrak" in tekst


@pytest.mark.parametrize("codering", ["utf-8-sig", "cp1252"])
def test_csv_met_bom_of_windows_codering(model, codering):
    """Zoals de app inleest: eerst utf-8-sig, dan cp1252."""
    tabel = pd.DataFrame([dict(TYPISCH, naam="Café")])[list(TYPISCH) + ["naam"]]
    bestand = tabel.to_csv(index=False, sep=";", decimal=",").encode(codering)
    for poging in ["utf-8-sig", "cp1252"]:
        try:
            terug = pd.read_csv(io.BytesIO(bestand), sep=None, engine="python", dtype=str, encoding=poging)
            break
        except UnicodeDecodeError:
            continue
    terug.columns = [c.replace("\ufeff", "").strip() for c in terug.columns]
    assert list(terug.columns)[0] == "pct_meergezinswoning" and terug.loc[0, "naam"] == "Café"
    assert voorspel_batch(terug, *model)["voorspelling_kwh"].notna().all()


@pytest.mark.parametrize("antwoord", [{"response": None}, [], "hoi", {"response": {"docs": None}},
                                      {"response": {"docs": ["x", 3, {"buurtcode": 5}]}}])
def test_postcode_vreemd_antwoord_geeft_geen_crash(antwoord):
    assert postcode.lees_antwoord(antwoord) == []


def test_uitleg_hoort_bij_dit_model(model):
    from predict import laad_combinatiecheck, pipeline_vingerafdruk
    assert lees_json("reports/feature_importance.json")["pipeline_sha256"] == pipeline_vingerafdruk()
    assert laad_combinatiecheck(model[1]) is not None
