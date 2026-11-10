"""
Stap 3: verkenning (alleen trainingsdata) en gemeentevergelijking.

- Grafieken over buurten gebruiken ALLEEN de trainingsbuurten, zodat de
  testgroep geen invloed heeft op onze keuzes.
- De gemeentevergelijking gebruikt de cijfers die het CBS zelf per gemeente
  publiceert (niet zelf berekend uit buurtgemiddelden).

Schrijft: reports/figures/*.png en reports/exploration_summary.json
Uitvoeren: python src/explore.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # grafieken opslaan zonder scherm
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOEL = "electricity_delivery_kwh_per_dwelling_year"
FIG_DIR = os.path.join(PROJECT_DIR, "reports", "figures")

LABELS = {
    "pct_meergezinswoning": "Meergezinswoningen (%)",
    "pct_bouwjaar_vanaf_2000": "Woningen gebouwd vanaf 2000 (%)",
    "gem_huishoudensgrootte": "Gemiddelde huishoudensgrootte (personen)",
    "stedelijkheid": "Stedelijkheid (1 = zeer sterk, 5 = niet stedelijk)",
    "woz_waarde_x1000": "Gemiddelde WOZ-waarde (x 1.000 euro)",
    "pct_koopwoningen": "Koopwoningen (%)",
    "huishoudens_totaal": "Aantal huishoudens in de buurt",
}
STEDELIJKHEID_NAMEN = {1: "zeer sterk", 2: "sterk", 3: "matig", 4: "weinig", 5: "niet"}


def pad(relatief_pad):
    return os.path.join(PROJECT_DIR, relatief_pad)


def opslaan(fig, naam):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, naam), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    tabel = pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})
    split = pd.read_csv(pad("reports/split_membership.csv"), dtype={"buurt_code": str})
    train = tabel.merge(split[["buurt_code", "set"]], on="buurt_code", how="inner", validate="one_to_one")
    train = train[train["set"] == "train"]

    # 1. Verdeling van de doelvariabele
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train[DOEL], bins=50, color="#2b6f9e")
    ax.axvline(train[DOEL].median(), color="black", linestyle="--",
               label=f"mediaan = {train[DOEL].median():,.0f} kWh".replace(",", "."))
    ax.set_xlabel("Gemiddelde elektriciteitslevering per woning (kWh per jaar, 2023)")
    ax.set_ylabel("Aantal buurten")
    ax.set_title("Verdeling doelvariabele (trainingsbuurten)")
    ax.legend()
    opslaan(fig, "01_verdeling_doel.png")

    # 2. Spreidingsgrafieken: kenmerk tegen doel
    kenmerken = ["pct_meergezinswoning", "pct_bouwjaar_vanaf_2000", "gem_huishoudensgrootte",
                 "woz_waarde_x1000", "pct_koopwoningen", "huishoudens_totaal"]
    fig, assen = plt.subplots(2, 3, figsize=(12, 7))
    for ax, kenmerk in zip(assen.ravel(), kenmerken):
        ax.scatter(train[kenmerk], train[DOEL], s=3, alpha=0.25, color="#2b6f9e")
        ax.set_xlabel(LABELS[kenmerk])
        ax.set_ylabel("kWh per woning")
    fig.suptitle("Buurtkenmerken en elektriciteitslevering (trainingsbuurten, 2023)")
    opslaan(fig, "02_spreiding_kenmerken.png")

    # 3. Vergelijking naar stedelijkheid
    fig, ax = plt.subplots(figsize=(7, 4))
    groepen = [train.loc[train["stedelijkheid"] == s, DOEL] for s in range(1, 6)]
    ax.boxplot(groepen, tick_labels=[f"{s}\n{STEDELIJKHEID_NAMEN[s]}" for s in range(1, 6)], showfliers=False)
    ax.set_xlabel("Stedelijkheid")
    ax.set_ylabel("kWh per woning per jaar")
    ax.set_title("Elektriciteitslevering naar stedelijkheid (trainingsbuurten)")
    opslaan(fig, "03_stedelijkheid.png")

    # 4. Vergelijking naar woningmix (klassen meergezinswoningen)
    klassen = pd.cut(train["pct_meergezinswoning"], bins=[-0.1, 10, 30, 50, 70, 90, 100],
                     labels=["0-10", "10-30", "30-50", "50-70", "70-90", "90-100"])
    fig, ax = plt.subplots(figsize=(7, 4))
    data_per_klasse = [train.loc[klassen == k, DOEL].dropna() for k in klassen.cat.categories]
    ax.boxplot(data_per_klasse, tick_labels=list(klassen.cat.categories), showfliers=False)
    ax.set_xlabel("Aandeel meergezinswoningen (%)")
    ax.set_ylabel("kWh per woning per jaar")
    ax.set_title("Elektriciteitslevering naar woningmix (trainingsbuurten)")
    opslaan(fig, "04_woningmix.png")

    # 5. Ontbrekende waarden (trainingsbuurten)
    alle = ["pct_meergezinswoning", "pct_bouwjaar_vanaf_2000", "gem_huishoudensgrootte", "stedelijkheid",
            "woz_waarde_x1000", "pct_koopwoningen"]
    missend = (train[alle].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([LABELS[k] for k in missend.index], missend.values, color="#c0504d")
    ax.set_xlabel("Ontbrekend (%)")
    ax.set_title("Ontbrekende waarden per kenmerk (trainingsbuurten)")
    opslaan(fig, "05_ontbrekend.png")

    # 6. Gemeentevergelijking met door CBS gepubliceerde gemeentecijfers
    gemeenten = pd.read_csv(pad("data/processed/gemeente_table.csv")).dropna(subset=[DOEL])
    gesorteerd = gemeenten.sort_values(DOEL)
    laag, hoog = gesorteerd.head(10), gesorteerd.tail(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    samen = pd.concat([laag, hoog])
    kleuren = ["#4f81bd"] * 10 + ["#c0504d"] * 10
    ax.barh(samen["gemeente_naam"], samen[DOEL], color=kleuren)
    ax.set_xlabel("Gemiddelde elektriciteitslevering per woning (kWh, 2023)")
    ax.set_title("10 laagste en 10 hoogste gemeenten (CBS-gemeentecijfers)")
    opslaan(fig, "06_gemeenten_laag_hoog.png")

    # Samenvatting in getallen (Spearman: rangcorrelatie, robuust voor scheve verdelingen)
    correlaties = {}
    for kenmerk in kenmerken + ["stedelijkheid"]:
        paar = train[[kenmerk, DOEL]].dropna()
        correlaties[kenmerk] = {
            "spearman": round(float(paar[kenmerk].corr(paar[DOEL], method="spearman")), 3),
            "pearson": round(float(paar[kenmerk].corr(paar[DOEL])), 3),
            "n": int(len(paar)),
        }
    samenvatting = {
        "basis": "Alleen trainingsbuurten (zie reports/split_membership.csv), behalve de gemeentevergelijking.",
        "n_trainingsbuurten": int(len(train)),
        "doel_mediaan_kwh": float(train[DOEL].median()),
        "doel_gemiddelde_kwh": round(float(train[DOEL].mean()), 1),
        "doel_p10_p90_kwh": [float(train[DOEL].quantile(0.1)), float(train[DOEL].quantile(0.9))],
        "correlaties_met_doel": correlaties,
        "mediaan_doel_per_stedelijkheid": {int(k): float(v) for k, v in train.groupby("stedelijkheid")[DOEL].median().items()},
        "mediaan_doel_per_meergezinsklasse": {str(k): float(v) for k, v in train.groupby(klassen, observed=True)[DOEL].median().items()},
        "gemeenten_cbs": {
            "aantal_met_cijfer": int(len(gemeenten)),
            "laagste_10": laag[["gemeente_naam", DOEL]].to_dict("records"),
            "hoogste_10": hoog[["gemeente_naam", DOEL]].to_dict("records"),
            "mediaan_kwh": float(gemeenten[DOEL].median()),
        },
    }
    with open(pad("reports/exploration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(samenvatting, f, ensure_ascii=False, indent=2)
    print("Geschreven: reports/figures/01..06 en reports/exploration_summary.json")


if __name__ == "__main__":
    main()
