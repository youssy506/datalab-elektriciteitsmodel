"""
Stap 6: het opgeslagen model uitleggen en controleren (na src/train.py).

Het model zelf verandert hier NIET. Dit script maakt:
1. Kenmerkbelang over alle testbuurten (permutation importance): hoeveel stijgt de test-MAE
   als je één kenmerk door elkaar husselt?            -> reports/feature_importance.json
2. Een controle op ongewone COMBINATIES van kenmerken (Local Outlier Factor, novelty-modus),
   geleerd op de trainingsbuurten en daarna gecontroleerd op de testbuurten
                                                       -> artifacts/combinatiecheck.joblib
                                                          reports/combinatiecheck.json
3. De modelkaart                                       -> MODEL_CARD.md

Uitvoeren: python src/explain.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor

import model_card
from predict import indicator_kenmerken, laad_model, pipeline_vingerafdruk

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AANDEEL_ONGEWOON = 0.01   # de 1% meest afwijkende trainingsbuurten bepaalt de grens
BUREN = 50                # aantal buren voor Local Outlier Factor
SEED = 42


def pad(relatief_pad):
    return os.path.join(PROJECT_DIR, relatief_pad)


def main():
    with open(pad("config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(pad("reports/metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    pipeline, metadata = laad_model()
    kenmerken = metadata["kenmerken"]
    doel = metadata["doel"]["interne_naam"]

    tabel = pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})
    split = pd.read_csv(pad("reports/split_membership.csv"), dtype={"buurt_code": str})
    data = tabel.merge(split[["buurt_code", "set"]], on="buurt_code", how="inner", validate="one_to_one")
    train = data[data["set"] == "train"].reset_index(drop=True)
    test = data[data["set"] == "test"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Kenmerkbelang op de testgemeenten
    # ------------------------------------------------------------------
    # Een kolom husselen = het verband tussen dat kenmerk en de uitkomst verbreken.
    # Let op: kenmerken die samenhangen (WOZ, koopwoningen, meergezinswoningen) kunnen elkaar
    # deels vervangen; hun belang wordt dan te laag ingeschat (zie de scikit-learn-documentatie).
    resultaat = permutation_importance(pipeline, test[kenmerken], test[doel], scoring="neg_mean_absolute_error",
                                       n_repeats=10, random_state=SEED)
    belang = {
        "methode": "Permutation importance op de testgemeenten (scikit-learn), 10 herhalingen, maat = toename MAE",
        "test_mae_kwh": float(mean_absolute_error(test[doel], pipeline.predict(test[kenmerken]))),
        "kenmerken": {k: {"toename_mae_gemiddeld": float(m), "toename_mae_sd": float(s)}
                      for k, m, s in zip(kenmerken, resultaat.importances_mean, resultaat.importances_std)},
        "let_op": "Samenhangende kenmerken kunnen elkaars belang verbergen. Belang is samenhang, geen oorzaak.",
        "model_versie": metadata["model_versie"],
        "pipeline_sha256": pipeline_vingerafdruk(),
    }
    with open(pad("reports/feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(belang, f, ensure_ascii=False, indent=2)
    for k, v in sorted(belang["kenmerken"].items(), key=lambda kv: -kv[1]["toename_mae_gemiddeld"]):
        print(f"  belang {k:26s} +{v['toename_mae_gemiddeld']:6.1f} kWh (sd {v['toename_mae_sd']:4.1f})")

    # ------------------------------------------------------------------
    # 2. Controle op ongewone combinaties
    # ------------------------------------------------------------------
    # Alleen buurten waarvan alle kenmerken bekend zijn; dezelfde schaal als het model gebruikt.
    def geschaald(deel):
        return pipeline[:-1].transform(deel[kenmerken])[:, :len(kenmerken)]

    train_compleet = train[train[kenmerken].notna().all(axis=1)]
    detector = LocalOutlierFactor(n_neighbors=BUREN, novelty=True, contamination=AANDEEL_ONGEWOON)
    detector.fit(geschaald(train_compleet))
    train_vlag = detector.predict(geschaald(train_compleet)) == -1

    test_compleet = test[test[kenmerken].notna().all(axis=1)].copy()
    test_compleet["ongewoon"] = detector.predict(geschaald(test_compleet)) == -1
    test_compleet["fout"] = (pipeline.predict(test_compleet[kenmerken]) - test_compleet[doel]).abs()
    gemarkeerd = test_compleet[test_compleet["ongewoon"]]
    overig = test_compleet[~test_compleet["ongewoon"]]
    validatie = {
        "methode": f"LocalOutlierFactor(n_neighbors={BUREN}, novelty=True, contamination={AANDEEL_ONGEWOON}) "
                   "op de geschaalde kenmerken van trainingsbuurten met alle kenmerken bekend",
        "grens_pct": 100 * AANDEEL_ONGEWOON,   # de grens: zo'n deel van de trainingsbuurten valt erbuiten
        "alleen_als": "alle kenmerken bekend zijn",
        "train_n_compleet": int(len(train_compleet)),
        "aandeel_training_gemarkeerd_pct": round(100 * float(train_vlag.mean()), 1),
        "test_n_compleet": int(len(test_compleet)),
        "test_n_gemarkeerd": int(len(gemarkeerd)),
        "test_aandeel_gemarkeerd_pct": round(100 * len(gemarkeerd) / max(len(test_compleet), 1), 1),
        "test_mae_gemarkeerd": float(gemarkeerd["fout"].mean()) if len(gemarkeerd) else None,
        "test_mae_overig": float(overig["fout"].mean()),
        "model_versie": metadata["model_versie"],
        "pipeline_sha256": pipeline_vingerafdruk(),
    }
    joblib.dump({"detector": detector, "model_versie": metadata["model_versie"], "kenmerken": kenmerken,
                 "pipeline_sha256": pipeline_vingerafdruk(), "validatie": validatie},
                pad("artifacts/combinatiecheck.joblib"))
    with open(pad("reports/combinatiecheck.json"), "w", encoding="utf-8") as f:
        json.dump(validatie, f, ensure_ascii=False, indent=2)
    print(f"  combinatiecheck: {validatie['test_n_gemarkeerd']} van {validatie['test_n_compleet']} testbuurten "
          f"gemarkeerd; MAE {validatie['test_mae_gemarkeerd'] or float('nan'):.0f} vs {validatie['test_mae_overig']:.0f} kWh")

    # ------------------------------------------------------------------
    # 3. Modelkaart
    # ------------------------------------------------------------------
    with open(pad("MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(model_card.markdown(metadata, metrics, config, belang, validatie,
                                    indicator_kenmerken(pipeline, metadata)))
    print("Geschreven: reports/feature_importance.json, artifacts/combinatiecheck.joblib, "
          "reports/combinatiecheck.json, MODEL_CARD.md")


if __name__ == "__main__":
    main()
