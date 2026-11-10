"""
Acceptatietests: bewijs dat de afspraken uit de specificatie zijn nagekomen.

Uitvoeren (na prepare_data, split_data en train):
    .venv/bin/python -m pytest -v
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from predict import laad_model, voorspel, voorspel_batch  # noqa: E402


def pad(p):
    return os.path.join(PROJECT_DIR, p)


@pytest.fixture(scope="module")
def config():
    with open(pad("config.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def modeltabel():
    return pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})


@pytest.fixture(scope="module")
def split():
    return pd.read_csv(pad("reports/split_membership.csv"), dtype={"buurt_code": str})


@pytest.fixture(scope="module")
def model():
    return laad_model()


# 1. Doeldefinitie komt overeen met de bronmetadata
def test_doeldefinitie_volgt_metadata(config):
    with open(pad(config["metadata_file"]), encoding="utf-8") as f:
        props = json.load(f)["value"]  # lijst; themagroepen hebben geen eigen Key
    doel = next(p for p in props if p.get("Key") == config["target_raw_column"])
    assert doel["Unit"] == "kWh"
    groep = next(p for p in props if p.get("ID") == doel["ParentID"])
    assert "individuele aansluitingen van particuliere woningen" in groep["Description"]
    assert "Collectieve" in groep["Description"]


# 2. Unieke buurt-jaar-sleutels, alleen buurten
def test_unieke_sleutels(modeltabel):
    assert not modeltabel.duplicated(["buurt_code", "target_year"]).any()
    assert modeltabel["buurt_code"].str.startswith("BU").all()


# 3. Geen joins met rijvermenigvuldiging (modeltabel = precies alle buurten)
def test_geen_rijvermenigvuldiging(modeltabel):
    with open(pad("reports/data_quality.json"), encoding="utf-8") as f:
        kwaliteit = json.load(f)
    assert len(modeltabel) == kwaliteit["buurten"] == kwaliteit["rijen_per_regiosoort"]["Buurt"]


# 4. Ontbrekende doelen niet geïmputeerd
def test_doel_niet_geimputeerd(modeltabel, split, config):
    doel = config["target_internal_name"]
    assert modeltabel.loc[~modeltabel["has_label"], doel].isna().all()
    # Alleen buurten mét CBS-waarde komen in train/test
    labels = set(modeltabel.loc[modeltabel["has_label"], "buurt_code"])
    assert set(split["buurt_code"]) == labels


# 5. Gemeenten blijven gescheiden (train/test én binnen de folds)
def test_groepen_gescheiden(split):
    train_gm = set(split.loc[split["set"] == "train", "gemeente_code"])
    test_gm = set(split.loc[split["set"] == "test", "gemeente_code"])
    assert train_gm.isdisjoint(test_gm)
    train = split[split["set"] == "train"]
    assert (train.groupby("gemeente_code")["cv_fold"].nunique() == 1).all()
    assert (split.loc[split["set"] == "test", "cv_fold"] == -1).all()


# 6. Voorbereiding alleen op trainingsdata geleerd (invulwaarden = mediaan training)
def test_imputatie_geleerd_op_training(model, modeltabel, split):
    pipeline, metadata = model
    train_codes = set(split.loc[split["set"] == "train", "buurt_code"])
    train = modeltabel[modeltabel["buurt_code"].isin(train_codes)]
    verwacht = train[metadata["kenmerken"]].median().to_numpy()
    geleerd = pipeline.named_steps["invullen"].statistics_
    assert np.allclose(verwacht, geleerd)


# 7. Alle modellen gebruiken dezelfde evaluatiepartities
def test_zelfde_folds_voor_alle_modellen():
    with open(pad("reports/metrics.json"), encoding="utf-8") as f:
        cv = json.load(f)["cross_validatie"]
    indelingen = {naam: [(r["fold"], r["n_buurten"]) for r in v["per_fold"]] for naam, v in cv.items()}
    eerste = next(iter(indelingen.values()))
    assert all(ind == eerste for ind in indelingen.values())


# 8. Prestatiematen reproduceerbaar uit opgeslagen testvoorspellingen
def test_metrics_reproduceerbaar():
    tp = pd.read_csv(pad("reports/test_predictions.csv"))
    with open(pad("reports/metrics.json"), encoding="utf-8") as f:
        m = json.load(f)["test_gekozen_model"]
    assert mean_absolute_error(tp["werkelijk_kwh"], tp["voorspeld_kwh"]) == pytest.approx(m["MAE"])
    assert np.sqrt(mean_squared_error(tp["werkelijk_kwh"], tp["voorspeld_kwh"])) == pytest.approx(m["RMSE"])
    assert r2_score(tp["werkelijk_kwh"], tp["voorspeld_kwh"]) == pytest.approx(m["R2"])


# 9. Opnieuw geladen model geeft dezelfde voorspellingen
def test_herladen_model(model, config):
    pipeline, metadata = model
    tp = pd.read_csv(pad("reports/test_predictions.csv"))
    tabel = pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})
    tabel = tabel.set_index("buurt_code").loc[tp["buurt_code"]]
    opnieuw = joblib.load(pad("artifacts/pipeline.joblib")).predict(tabel[metadata["kenmerken"]])
    assert np.max(np.abs(opnieuw - tp["voorspeld_kwh"].to_numpy())) <= config["reload_tolerance_kwh"]


# 10. De applicatie/voorspelroute behandelt geldige, ontbrekende, ongeldige en afwijkende invoer
GELDIG = {"pct_meergezinswoning": 20, "pct_bouwjaar_vanaf_2000": 10, "gem_huishoudensgrootte": 2.3,
          "stedelijkheid": 3, "woz_waarde_x1000": 350, "pct_koopwoningen": 60}


def test_geldige_invoer(model):
    r = voorspel(dict(GELDIG), *model)
    assert r["fouten"] == [] and r["waarschuwingen"] == []
    assert 500 < r["voorspelling_kwh"] < 10000


def test_ontbrekende_invoer_geeft_waarschuwing(model):
    invoer = dict(GELDIG, woz_waarde_x1000=None)
    r = voorspel(invoer, *model)
    assert r["voorspelling_kwh"] is not None
    assert any("niet ingevuld" in w for w in r["waarschuwingen"])


@pytest.mark.parametrize("kenmerk,waarde", [
    ("pct_meergezinswoning", 150),        # percentage boven 100
    ("pct_koopwoningen", -5),             # negatief percentage
    ("stedelijkheid", 2.5),               # geen geheel getal
    ("gem_huishoudensgrootte", "twee"),   # geen getal
    ("woz_waarde_x1000", float("inf")),   # niet eindig
])
def test_ongeldige_invoer_geeft_fout(model, kenmerk, waarde):
    r = voorspel(dict(GELDIG, **{kenmerk: waarde}), *model)
    assert r["voorspelling_kwh"] is None and r["fouten"]


def test_onbekend_en_ontbrekend_kenmerk_geeft_fout(model):
    zonder = {k: v for k, v in GELDIG.items() if k != "stedelijkheid"}
    assert voorspel(zonder, *model)["fouten"]
    assert voorspel(dict(GELDIG, inkomen=40), *model)["fouten"]


def test_afwijkende_invoer_geeft_waarschuwing(model):
    r = voorspel(dict(GELDIG, woz_waarde_x1000=3000), *model)  # geldig, maar buiten trainingsbereik
    assert r["voorspelling_kwh"] is not None
    assert any("buiten het bereik" in w for w in r["waarschuwingen"])


def test_batch_met_leeg_bestand(model):
    leeg = pd.DataFrame(columns=list(GELDIG))
    uit = voorspel_batch(leeg, *model)
    assert len(uit) == 0 and "voorspelling_kwh" in uit.columns


def test_leeg_veld_is_niet_hetzelfde_als_mediaan(model):
    """Het model gebruikt ook 'ontbreekt'-kolommen; de tekst in app en waarschuwing moet dat zeggen."""
    pipeline, metadata = model
    r = voorspel(dict(GELDIG, woz_waarde_x1000=None), *model)
    assert any("niet gelijk aan het invullen van de mediaan" in w for w in r["waarschuwingen"])


def test_batch_gebruikt_zelfde_route(model):
    tabel = pd.DataFrame([GELDIG, dict(GELDIG, pct_meergezinswoning=150)])
    uit = voorspel_batch(tabel, *model)
    assert uit.loc[0, "voorspelling_kwh"] == pytest.approx(voorspel(dict(GELDIG), *model)["voorspelling_kwh"])
    assert pd.isna(uit.loc[1, "voorspelling_kwh"]) and uit.loc[1, "fouten"]
