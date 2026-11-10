"""
Stap 4: modellen vergelijken, het gekozen model testen en opslaan.

Werkwijze
1. Lees de modeltabel en de vaste indeling in train/test en folds.
2. Vergelijk kandidaat-modellen met cross-validatie op de TRAININGSgemeenten
   (alle modellen gebruiken dezelfde folds):
     - mediaanbaseline (voorspelt altijd de mediaan van de training)
     - Ridge-regressie (lineair, goed uit te leggen) met basis- en uitgebreide kenmerken
     - beperkt Random Forest met basis- en uitgebreide kenmerken
3. Kies het model met de laagste gemiddelde MAE in de cross-validatie.
4. Train dat model op alle trainingsbuurten en evalueer het ÉÉN keer op de testgemeenten.
5. Sla de complete pipeline (invullen + schalen + model) op, plus metadata en resultaten.

Uitvoeren: python src/train.py
"""

import json
import os
import platform
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_DIR, "reports", "figures")


def pad(relatief_pad):
    return os.path.join(PROJECT_DIR, relatief_pad)


def maak_pipeline(model):
    """Voorbereiding en model in één object, zodat alles alleen op trainingsdata wordt geleerd.

    - SimpleImputer: vult ontbrekende KENMERKEN in met de mediaan van de trainingsdata
      en voegt een 0/1-kolom toe die aangeeft dat er iets ontbrak.
    - StandardScaler: zet kenmerken op dezelfde schaal (nodig voor Ridge).
    """
    return Pipeline([
        ("invullen", SimpleImputer(strategy="median", add_indicator=True)),
        ("schalen", StandardScaler()),
        ("model", model),
    ])


def foutmaten(werkelijk, voorspeld):
    return {
        "MAE": float(mean_absolute_error(werkelijk, voorspeld)),
        "RMSE": float(np.sqrt(mean_squared_error(werkelijk, voorspeld))),
        "R2": float(r2_score(werkelijk, voorspeld)),
    }


def main():
    with open(pad("config.json"), encoding="utf-8") as f:
        config = json.load(f)
    doel = config["target_internal_name"]
    os.makedirs(pad("artifacts"), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data en vaste indeling inlezen
    # ------------------------------------------------------------------
    tabel = pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})
    split = pd.read_csv(pad("reports/split_membership.csv"), dtype={"buurt_code": str})
    data = tabel.merge(split[["buurt_code", "set", "cv_fold"]], on="buurt_code",
                       how="inner", validate="one_to_one")
    if data[doel].isna().any():
        raise SystemExit("Er staan rijen zonder doelwaarde in de modeldata.")

    train = data[data["set"] == "train"].reset_index(drop=True)
    test = data[data["set"] == "test"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Kandidaten
    # ------------------------------------------------------------------
    # Elke kandidaat = (welke kenmerkenset, functie die een nieuw, ongetraind model maakt)
    rf = config["models"]["random_forest"]
    alpha = config["models"]["ridge_alpha"]

    def nieuwe_baseline():
        return maak_pipeline(DummyRegressor(strategy="median"))

    def nieuwe_ridge():
        return maak_pipeline(Ridge(alpha=alpha))

    def nieuw_forest():
        return maak_pipeline(RandomForestRegressor(**rf))

    kandidaten = {"mediaanbaseline": ("basis", nieuwe_baseline)}
    for set_naam in config["feature_sets"]:
        kandidaten[f"ridge_{set_naam}"] = (set_naam, nieuwe_ridge)
        kandidaten[f"random_forest_{set_naam}"] = (set_naam, nieuw_forest)

    # ------------------------------------------------------------------
    # 3. Cross-validatie met dezelfde folds voor ieder model
    # ------------------------------------------------------------------
    folds = sorted(train["cv_fold"].unique())
    cv_resultaten = {}
    for naam, (set_naam, bouw_model) in kandidaten.items():
        kenmerken = config["feature_sets"][set_naam]
        per_fold = []
        for fold in folds:
            fit_deel = train[train["cv_fold"] != fold]
            val_deel = train[train["cv_fold"] == fold]
            model = bouw_model()
            model.fit(fit_deel[kenmerken], fit_deel[doel])
            maten = foutmaten(val_deel[doel], model.predict(val_deel[kenmerken]))
            maten.update({"fold": int(fold), "n_buurten": int(len(val_deel)),
                          "n_gemeenten": int(val_deel["gemeente_code"].nunique())})
            per_fold.append(maten)
        df = pd.DataFrame(per_fold)
        cv_resultaten[naam] = {
            "kenmerkenset": set_naam,
            "kenmerken": kenmerken,
            "MAE_gemiddeld": float(df["MAE"].mean()), "MAE_sd": float(df["MAE"].std()),
            "RMSE_gemiddeld": float(df["RMSE"].mean()), "RMSE_sd": float(df["RMSE"].std()),
            "R2_gemiddeld": float(df["R2"].mean()), "R2_sd": float(df["R2"].std()),
            "per_fold": per_fold,
        }
        print(f"{naam:28s} CV-MAE {df['MAE'].mean():7.1f} (sd {df['MAE'].std():5.1f})  "
              f"RMSE {df['RMSE'].mean():7.1f}  R2 {df['R2'].mean():.3f}")

    # ------------------------------------------------------------------
    # 4. Kiezen: laagste gemiddelde CV-MAE (baseline doet niet mee als keuze)
    # ------------------------------------------------------------------
    keuze = min((n for n in cv_resultaten if n != "mediaanbaseline"),
                key=lambda n: cv_resultaten[n]["MAE_gemiddeld"])
    keuze_set, keuze_bouw = kandidaten[keuze]
    kenmerken = config["feature_sets"][keuze_set]
    print(f"\nGekozen model: {keuze}  (kenmerken: {kenmerken})")

    # ------------------------------------------------------------------
    # 5. Trainen op alle trainingsbuurten, één keer testen
    # ------------------------------------------------------------------
    pipeline = keuze_bouw()
    pipeline.fit(train[kenmerken], train[doel])
    test_voorspelling = pipeline.predict(test[kenmerken])
    test_maten = foutmaten(test[doel], test_voorspelling)

    baseline = maak_pipeline(DummyRegressor(strategy="median")).fit(train[kenmerken], train[doel])
    baseline_maten = foutmaten(test[doel], baseline.predict(test[kenmerken]))

    print(f"Test (gekozen):  MAE {test_maten['MAE']:.1f}  RMSE {test_maten['RMSE']:.1f}  R2 {test_maten['R2']:.3f}")
    print(f"Test (baseline): MAE {baseline_maten['MAE']:.1f}  RMSE {baseline_maten['RMSE']:.1f}  R2 {baseline_maten['R2']:.3f}")

    # Testvoorspellingen opslaan (hieruit kun je alle testmaten opnieuw berekenen)
    extra = [k for k in kenmerken if k != "stedelijkheid"]  # stedelijkheid niet dubbel opnemen
    test_uit = test[["buurt_code", "buurt_naam", "gemeente_code", "gemeente_naam", "stedelijkheid"] + extra].copy()
    test_uit["kenmerk_ontbreekt"] = test[kenmerken].isna().any(axis=1)
    test_uit["werkelijk_kwh"] = test[doel]
    test_uit["voorspeld_kwh"] = test_voorspelling
    test_uit["baseline_kwh"] = baseline.predict(test[kenmerken])
    test_uit["residu_kwh"] = test_uit["werkelijk_kwh"] - test_uit["voorspeld_kwh"]
    test_uit.to_csv(pad("reports/test_predictions.csv"), index=False)

    # Fouten per subgroep (alleen groepen met minstens 30 testbuurten)
    subgroepen = {}
    for kolom in ["stedelijkheid", "kenmerk_ontbreekt"]:
        resultaat = {}
        for waarde, groep in test_uit.groupby(kolom):
            if len(groep) >= 30:
                resultaat[str(waarde)] = {"n": int(len(groep)),
                                          **foutmaten(groep["werkelijk_kwh"], groep["voorspeld_kwh"])}
            else:
                resultaat[str(waarde)] = {"n": int(len(groep)), "opmerking": "te weinig buurten voor een betrouwbare foutmaat"}
        subgroepen[kolom] = resultaat

    # ------------------------------------------------------------------
    # 6. Opslaan en controleren dat het opnieuw geladen model hetzelfde doet
    # ------------------------------------------------------------------
    joblib.dump(pipeline, pad("artifacts/pipeline.joblib"))
    herladen = joblib.load(pad("artifacts/pipeline.joblib"))
    max_verschil = float(np.max(np.abs(herladen.predict(test[kenmerken]) - test_voorspelling)))
    herlaad_ok = max_verschil <= config["reload_tolerance_kwh"]

    # Bereik van de trainingsdata per kenmerk (voor waarschuwingen in de app)
    bereik = {}
    for k in kenmerken:
        w = train[k].dropna()
        bereik[k] = {"min": float(w.min()), "max": float(w.max()),
                     "p01": float(w.quantile(0.01)), "p99": float(w.quantile(0.99)),
                     "mediaan": float(w.median())}

    # Uitleg van het lineaire model (Ridge met dezelfde kenmerken, getraind op trainingsdata)
    ridge_naam = f"ridge_{keuze_set}"
    ridge = kandidaten[ridge_naam][1]().fit(train[kenmerken], train[doel])
    imputer = ridge.named_steps["invullen"]
    kolomnamen = list(kenmerken) + [f"ontbreekt_{kenmerken[i]}" for i in imputer.indicator_.features_]
    schaal = ridge.named_steps["schalen"].scale_
    coef = ridge.named_steps["model"].coef_
    ridge_uitleg = {
        "intercept_kwh": float(ridge.named_steps["model"].intercept_),
        "coefficienten": [
            {"kenmerk": n, "effect_per_standaarddeviatie_kwh": float(c), "standaarddeviatie": float(s),
             "effect_per_eenheid_kwh": float(c / s) if s > 0 else None}
            for n, c, s in zip(kolomnamen, coef, schaal)
        ],
        "let_op": "Samenhang, geen oorzaak. Effect per eenheid geldt als de andere kenmerken gelijk blijven.",
    }

    nu = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "model_versie": f"{config['task_name']}-{datetime.now():%Y%m%d}",
        "getraind_op": nu,
        "gekozen_model": keuze,
        "kenmerken": kenmerken,
        "doel": {
            "interne_naam": doel, "ruwe_kolom": config["target_raw_column"],
            "definitie": config["target_definition"], "eenheid": config["target_unit"],
            "referentiejaar": config["target_year"], "geografisch_niveau": config["geographic_level"],
            "bron": f"CBS {config['source_table']} - {config['source_title']}",
            "wat_het_niet_is": [
                "Niet het volledige elektriciteitsverbruik achter de meter (eigen opwek telt niet volledig mee).",
                "Niet het totale energiegebruik van een buurt.",
                "Niet het verbruik van één huishouden.",
                "Niet het piekvermogen of de kans op netcongestie.",
            ],
        },
        "training": {"n_buurten": int(len(train)), "n_gemeenten": int(train["gemeente_code"].nunique())},
        "test": {"n_buurten": int(len(test)), "n_gemeenten": int(test["gemeente_code"].nunique()),
                 **test_maten},
        "baseline_test": baseline_maten,
        "kenmerk_bereik_training": bereik,
        "ridge_uitleg": ridge_uitleg,
        "herlaadcontrole": {"max_verschil_kwh": max_verschil, "tolerantie_kwh": config["reload_tolerance_kwh"],
                            "geslaagd": bool(herlaad_ok)},
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    with open(pad("artifacts/model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    metrics = {
        "primaire_foutmaat": "Ongewogen MAE over buurten (kWh per woning per jaar)",
        "validatie": "GroupKFold binnen trainingsgemeenten; dezelfde folds voor alle modellen",
        "cross_validatie": cv_resultaten,
        "gekozen_model": keuze,
        "keuzeregel": "Laagste gemiddelde CV-MAE (mediaanbaseline is alleen referentie)",
        "test_gekozen_model": test_maten,
        "test_mediaanbaseline": baseline_maten,
        "test_verbetering_mae_tov_baseline_pct": round(100 * (1 - test_maten["MAE"] / baseline_maten["MAE"]), 1),
        "test_n_buurten": int(len(test)), "test_n_gemeenten": int(test["gemeente_code"].nunique()),
        "fouten_per_subgroep_test": subgroepen,
        "herlaadcontrole_geslaagd": bool(herlaad_ok),
    }
    with open(pad("reports/metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 7. Grafieken over de testresultaten
    # ------------------------------------------------------------------
    # Werkelijk tegen voorspeld
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(test_uit["werkelijk_kwh"], test_uit["voorspeld_kwh"], s=4, alpha=0.3, color="#2b6f9e")
    grens = [test_uit[["werkelijk_kwh", "voorspeld_kwh"]].min().min(), test_uit[["werkelijk_kwh", "voorspeld_kwh"]].max().max()]
    ax.plot(grens, grens, color="black", linestyle="--", label="perfecte voorspelling")
    ax.set_xlabel("Werkelijk (kWh per woning per jaar)")
    ax.set_ylabel("Voorspeld (kWh per woning per jaar)")
    ax.set_title(f"Testgemeenten: werkelijk vs. voorspeld\n({keuze})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "07_test_werkelijk_vs_voorspeld.png"), dpi=150); plt.close(fig)

    # Residuen
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(test_uit["voorspeld_kwh"], test_uit["residu_kwh"], s=4, alpha=0.3, color="#2b6f9e")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_xlabel("Voorspeld (kWh per woning per jaar)")
    ax.set_ylabel("Residu = werkelijk - voorspeld (kWh)")
    ax.set_title("Residuen in de testgemeenten")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "08_test_residuen.png"), dpi=150); plt.close(fig)

    # Modelvergelijking (CV-MAE met spreiding)
    namen = list(cv_resultaten)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(namen, [cv_resultaten[n]["MAE_gemiddeld"] for n in namen],
            xerr=[cv_resultaten[n]["MAE_sd"] for n in namen],
            color=["#c0504d" if n == keuze else "#7f7f7f" for n in namen])
    ax.set_xlabel("Gemiddelde MAE in cross-validatie (kWh per woning per jaar)")
    ax.set_title("Modelvergelijking (lager is beter; streepje = spreiding over folds)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "09_modelvergelijking_cv.png"), dpi=150); plt.close(fig)

    # Fout per stedelijkheidsklasse
    sted = {k: v for k, v in subgroepen["stedelijkheid"].items() if "MAE" in v}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([str(int(float(k))) for k in sted], [v["MAE"] for v in sted.values()], color="#2b6f9e")
    ax.set_xlabel("Stedelijkheid (1 = zeer sterk, 5 = niet stedelijk)")
    ax.set_ylabel("Test-MAE (kWh)")
    ax.set_title("Testfout per stedelijkheidsklasse")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "10_test_mae_per_stedelijkheid.png"), dpi=150); plt.close(fig)

    print(f"Herlaadcontrole: max verschil {max_verschil:.2e} kWh -> {'OK' if herlaad_ok else 'NIET OK'}")
    print("Geschreven: artifacts/pipeline.joblib, artifacts/model_metadata.json, reports/metrics.json, "
          "reports/test_predictions.csv, reports/figures/07..10")


if __name__ == "__main__":
    main()
