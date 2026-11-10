"""
Stap 2: testgroep reserveren (vóór verkenning en modelkeuze).

We leggen ongeveer 20% van de GEMEENTEN apart als testgroep. Alle buurten van
een testgemeente zitten dus samen in de test. Zo meten we hoe goed het model
werkt voor gemeenten die het nooit heeft gezien.

Daarnaast verdelen we de trainingsgemeenten over 5 folds (GroupKFold).
Alle modellen gebruiken later precies deze folds.

Schrijft: reports/split_membership.csv
Uitvoeren: python src/split_data.py
"""

import json
import os

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pad(relatief_pad):
    return os.path.join(PROJECT_DIR, relatief_pad)


def main():
    with open(pad("config.json"), encoding="utf-8") as f:
        config = json.load(f)
    instellingen = config["split"]

    tabel = pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})
    # Alleen buurten met een gepubliceerde doelwaarde doen mee.
    tabel = tabel[tabel["has_label"]].reset_index(drop=True)
    groepen = tabel[instellingen["group_column"]]

    # 1. Testgroep: ~20% van de gemeenten.
    splitter = GroupShuffleSplit(n_splits=1, test_size=instellingen["test_size_groups"],
                                 random_state=instellingen["random_state"])
    train_idx, test_idx = next(splitter.split(tabel, groups=groepen))
    tabel["set"] = "train"
    tabel.loc[test_idx, "set"] = "test"

    # 2. Folds binnen de trainingsgemeenten.
    tabel["cv_fold"] = -1
    train = tabel[tabel["set"] == "train"]
    folds = GroupKFold(n_splits=instellingen["n_cv_folds"])
    for fold_nummer, (_, val_idx) in enumerate(folds.split(train, groups=train[instellingen["group_column"]])):
        tabel.loc[train.index[val_idx], "cv_fold"] = fold_nummer

    # Controle: geen gemeente zit in zowel train als test.
    gemeenten_train = set(tabel.loc[tabel["set"] == "train", "gemeente_code"])
    gemeenten_test = set(tabel.loc[tabel["set"] == "test", "gemeente_code"])
    overlap = gemeenten_train & gemeenten_test
    if overlap:
        raise SystemExit(f"Gemeenten in train én test: {overlap}")

    kolommen = ["buurt_code", "target_year", "gemeente_code", "gemeente_naam", "set", "cv_fold"]
    tabel[kolommen].to_csv(pad("reports/split_membership.csv"), index=False)

    print(f"Train: {len(gemeenten_train)} gemeenten, {(tabel['set'] == 'train').sum()} buurten")
    print(f"Test:  {len(gemeenten_test)} gemeenten, {(tabel['set'] == 'test').sum()} buurten")
    print("Geschreven: reports/split_membership.csv")


if __name__ == "__main__":
    main()
