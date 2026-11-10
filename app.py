"""
Streamlit-app: schat de gemiddelde elektriciteitslevering per woning in een buurt.

Starten (vanuit de projectmap):
    .venv/bin/streamlit run app.py

Ontwerpkeuzes staan in reports/design/design_contract.md.
Alle voorspellingen lopen via src/predict.py (dezelfde route als de batchverwerking).
"""

import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import model_card  # noqa: E402
from postcode import PostcodeFout, zoek_buurten  # noqa: E402
from predict import (INVOER_SPEC, combinatie_testuitslag, indicator_kenmerken, laad_model, nl_getal,  # noqa: E402
                     pipeline_vingerafdruk, verklaar, voorspel, voorspel_batch)

# Kleuren (zie ontwerpcontract): petrol = model/CBS, amber = jouw schatting, grijsblauw = referentie
PETROL = "#0F5F5C"
AMBER = "#B07600"         # alleen voor grafische markeringen (3,6:1, voldoet voor vormen)
AMBER_TEKST = "#8A5A00"   # amber voor tekst (5,6:1)
AS_KLEUR = "#5A6472"      # aslabels (5,6:1)
REFERENTIE = "#5B7083"
EIGEN = "#1C2430"         # jouw eigen levering (donker vierkant)
BAND = "#D5DDE3"
# Vega-Lite formatteert als 1,500; dit maakt er 1.500 van (Nederlandse notatie)
NL_GETAL = "replace(format(datum.value, ',.0f'), ',', '.')"
NL_DECIMAAL = "replace(format(datum.value, '.1f'), '.', ',')"

STEDELIJKHEID_KORT = {1: "1 · zeer sterk", 2: "2 · sterk", 3: "3 · matig", 4: "4 · weinig", 5: "5 · niet"}
STEDELIJKHEID_LANG = {1: "zeer sterk stedelijk (≥ 2.500 adressen per km²)", 2: "sterk stedelijk (1.500–2.500)",
                      3: "matig stedelijk (1.000–1.500)", 4: "weinig stedelijk (500–1.000)",
                      5: "niet stedelijk (< 500 adressen per km²)"}
KORTE_LABELS = {
    "pct_meergezinswoning": "Meergezinswoningen",
    "pct_bouwjaar_vanaf_2000": "Gebouwd vanaf 2000",
    "gem_huishoudensgrootte": "Huishoudensgrootte",
    "stedelijkheid": "Stedelijkheid",
    "woz_waarde_x1000": "Gem. WOZ-waarde",
    "pct_koopwoningen": "Koopwoningen",
}
EENHEID_KORT = {"pct_meergezinswoning": "%", "pct_bouwjaar_vanaf_2000": "%", "gem_huishoudensgrootte": "personen",
                "stedelijkheid": "klasse", "woz_waarde_x1000": "× € 1.000", "pct_koopwoningen": "%"}
MAX_BATCH_RIJEN = 5000
WONING_VELDEN = ["pct_meergezinswoning", "pct_bouwjaar_vanaf_2000", "woz_waarde_x1000", "pct_koopwoningen"]

st.set_page_config(page_title="Elektriciteitslevering per woning · DataLab", page_icon="⚡", layout="wide")


# ----------------------------------------------------------------------
# Hulpfuncties
# ----------------------------------------------------------------------
def pad(relatief_pad):
    return os.path.join(PROJECT_DIR, relatief_pad)


def aantal(n):
    return f"{int(n):,}".replace(",", ".")


def kwh(waarde, eenheid=True):
    """Afronden op tientallen (zoals het CBS) in Nederlandse notatie."""
    tekst = aantal(round(float(waarde), -1))
    return f"{tekst} kWh" if eenheid else tekst


def toelichting(tekst):
    """Uitleg in grijze tekst die wél voldoet aan 4,5:1 (st.caption is te licht)."""
    st.markdown(f":gray[{tekst}]")


def stijl(grafiek):
    """Gedeelde grafiekstijl: geen kader, aslabels met voldoende contrast."""
    return grafiek.configure_view(stroke=None).configure_axis(
        labelColor=AS_KLEUR, titleColor=AS_KLEUR, domainColor="#C9C3B6", tickColor="#C9C3B6")


def csv_bytes(df):
    """CSV die Nederlandse Excel goed opent: puntkomma, decimale komma, UTF-8 met BOM."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def json_laden(relatief_pad):
    """Optioneel bestand (van src/explain.py); None als het ontbreekt."""
    if not os.path.exists(pad(relatief_pad)):
        return None
    with open(pad(relatief_pad), encoding="utf-8") as f:
        return json.load(f)


def percentiel(waarde, gesorteerd):
    """Welk deel van de trainingsbuurten heeft een lagere levering?"""
    return 100 * np.searchsorted(gesorteerd, waarde, side="left") / len(gesorteerd)


@st.cache_resource(show_spinner="Model en gegevens laden…")
def model_laden():
    return laad_model()


@st.cache_data(show_spinner="Model en gegevens laden…")
def tabellen_laden():
    tabel = pd.read_csv(pad("data/processed/model_table.csv"), dtype={"buurt_code": str})
    split = pd.read_csv(pad("reports/split_membership.csv"), dtype={"buurt_code": str})
    tabel = tabel.merge(split[["buurt_code", "set"]], on="buurt_code", how="left", validate="one_to_one")
    gemeenten = pd.read_csv(pad("data/processed/gemeente_table.csv"))
    testvoorspellingen = pd.read_csv(pad("reports/test_predictions.csv"), dtype={"buurt_code": str})
    with open(pad("reports/metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    regio = pd.read_csv(pad("data/processed/regio_kenmerken.csv"))
    return tabel, gemeenten, testvoorspellingen, metrics, regio


@st.cache_data(show_spinner=False)
def extra_laden():
    """Uitleg en controles van src/explain.py (optioneel) en de projectinstellingen."""
    with open(pad("config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return config, json_laden("reports/feature_importance.json"), json_laden("reports/combinatiecheck.json")


# ----------------------------------------------------------------------
# Laden (met duidelijke foutmelding als de modelbestanden ontbreken)
# ----------------------------------------------------------------------
try:
    pipeline, metadata = model_laden()
    tabel, gemeenten, testvoorspellingen, metrics, regio = tabellen_laden()
    config, belang, combinatie = extra_laden()
except Exception as fout:  # ontbrekend of beschadigd bestand: duidelijke melding i.p.v. een traceback
    st.title("Elektriciteitslevering per woning")
    bestand = os.path.basename(getattr(fout, "filename", None) or "") or type(fout).__name__
    st.error(f"Het model of de gegevens konden niet worden geladen ({bestand}). "
             "Voer de stappen uit de README (opnieuw) uit, vanuit de projectmap:")
    st.code("python src/prepare_data.py\npython src/split_data.py\npython src/explore.py\npython src/train.py\n"
            "python src/explain.py",
            language="bash")
    st.stop()

# Uitleg van een ánder (opnieuw getraind) model niet tonen: vergelijk de vingerafdruk van pipeline.joblib
VINGERAFDRUK = pipeline_vingerafdruk()
belang = belang if belang and belang.get("pipeline_sha256") == VINGERAFDRUK else None
combinatie = combinatie if combinatie and combinatie.get("pipeline_sha256") == VINGERAFDRUK else None
MET_INDICATOR = indicator_kenmerken(pipeline, metadata)
doel = metadata["doel"]
test = metadata["test"]
kenmerken = metadata["kenmerken"]
bereik = metadata["kenmerk_bereik_training"]
DOELKOLOM = doel["interne_naam"]
train = tabel[tabel["set"] == "train"]
train_doel = np.sort(train[DOELKOLOM].to_numpy())
REF = {
    "p10": float(np.percentile(train_doel, 10)), "p50": float(np.median(train_doel)),
    "p90": float(np.percentile(train_doel, 90)), "p01": float(np.percentile(train_doel, 1)),
    "p99": float(np.percentile(train_doel, 99)), "n": len(train_doel),
}
BANDUITLEG = (f"Grijze band: hier ligt 80% van de {aantal(REF['n'])} trainingsbuurten "
              f"({kwh(REF['p10'])} tot {kwh(REF['p90'])}). Dit is de spreiding tussen buurten, geen foutmarge.")


# ----------------------------------------------------------------------
# Grafieken
# ----------------------------------------------------------------------
def leveringsmeter(markeringen, hoogte=118):
    """Horizontale meter: waar valt een waarde tussen alle trainingsbuurten?

    markeringen: lijst met dicts {waarde, label, kleur, vorm}. De grijze band is de
    spreiding van buurten (P10-P90), GEEN onzekerheidsinterval van de schatting.
    Verticaal gebruiken we een verborgen schaal van 0 tot 1:
    labels boven (0,88), band en stippen (0,55), referentietekst onder (0,12).
    """
    waarden = [m["waarde"] for m in markeringen]
    laag = min([REF["p01"]] + waarden) - 150
    hoog = max([REF["p99"]] + waarden) + 150
    schaal_x = alt.Scale(domain=[laag, hoog], nice=False)
    schaal_y = alt.Scale(domain=[0, 1])
    x = alt.X("waarde:Q", scale=schaal_x,
              axis=alt.Axis(title="kWh per woning per jaar", labelExpr=NL_GETAL, tickCount=6, grid=False))

    def y(veld):
        return alt.Y(f"{veld}:Q", scale=schaal_y, axis=None)

    band = alt.Chart(pd.DataFrame({"van": [REF["p10"]], "tot": [REF["p90"]], "y1": [0.40], "y2": [0.70],
                                   "p10": [kwh(REF["p10"])], "p90": [kwh(REF["p90"])]})).mark_rect(
        color=BAND, cornerRadius=3).encode(
        x=alt.X("van:Q", scale=schaal_x), x2="tot:Q", y=y("y1"), y2="y2:Q",
        tooltip=[alt.Tooltip("p10:N", title="P10"), alt.Tooltip("p90:N", title="P90")])
    mediaan = alt.Chart(pd.DataFrame({"waarde": [REF["p50"]], "y1": [0.34], "y2": [0.76],
                                      "t": [kwh(REF["p50"])]})).mark_rule(color=REFERENTIE, strokeWidth=2).encode(
        x=x, y=y("y1"), y2="y2:Q", tooltip=[alt.Tooltip("t:N", title="Mediaan")])
    # Alleen het mediaanlabel staat in de grafiek; de band wordt uitgelegd in de toelichting (BANDUITLEG),
    # zodat labels ook op een smal scherm niet over elkaar vallen.
    referentie = alt.Chart(pd.DataFrame({"waarde": [REF["p50"]], "yt": [0.12],
                                         "t": [f"mediaan {kwh(REF['p50'], False)}"]})).mark_text(
        color=AS_KLEUR, fontSize=11, align="center").encode(x=x, y=y("yt"), text="t:N")

    lagen = [band, mediaan, referentie]
    if markeringen:
        df = pd.DataFrame(markeringen).assign(ym=0.55, yl=0.90)
        df["kwh"] = df["waarde"].map(kwh)
        df["tekstkleur"] = df["kleur"].replace({AMBER: AMBER_TEKST})
        tooltip = [alt.Tooltip("label:N", title=" "), alt.Tooltip("kwh:N", title="Waarde")]
        # Gevulde stip (schatting) eerst, open ruit (CBS) erbovenop: zo blijven beide zichtbaar als ze dicht bij elkaar liggen
        gevuld = df[df["vorm"] != "diamond"]
        open_ = df[df["vorm"] == "diamond"]
        if len(gevuld):
            lagen.append(alt.Chart(gevuld).mark_point(filled=True, size=240, opacity=1, stroke="white",
                                                      strokeWidth=1.5).encode(
                x=x, y=y("ym"), color=alt.Color("kleur:N", scale=None), shape=alt.Shape("vorm:N", scale=None),
                tooltip=tooltip))
        if len(open_):
            lagen.append(alt.Chart(open_).mark_point(filled=False, size=380, strokeWidth=2.5, opacity=1).encode(
                x=x, y=y("ym"), color=alt.Color("kleur:N", scale=None), shape=alt.Shape("vorm:N", scale=None),
                tooltip=tooltip))
        if len(markeringen) == 1:
            # Bij meerdere markeringen zou tekst overlappen; dan staat de uitleg in de toelichting.
            lagen.append(alt.Chart(df).mark_text(fontWeight="bold", fontSize=12).encode(
                x=x, y=y("yl"), text="label:N", color=alt.Color("tekstkleur:N", scale=None)))

    return stijl(alt.layer(*lagen).properties(height=hoogte))


def wat_als_grafiek(invoer, kenmerk, huidige_schatting):
    """Hoe verandert de schatting als je één kenmerk varieert (de rest blijft gelijk)?"""
    if kenmerk == "stedelijkheid":
        rooster = np.arange(1, 6)
    else:
        rooster = np.linspace(bereik[kenmerk]["p01"], bereik[kenmerk]["p99"], 40)
    basis = {k: (np.nan if invoer.get(k) is None else float(invoer[k])) for k in kenmerken}
    rijen = pd.DataFrame([dict(basis, **{kenmerk: float(w)}) for w in rooster], columns=kenmerken)
    df = pd.DataFrame({"waarde": rooster, "schatting": pipeline.predict(rijen)})
    df["waarde_t"] = df["waarde"].map(nl_getal)
    df["schatting_t"] = df["schatting"].map(kwh)
    titel = f"{KORTE_LABELS[kenmerk]} ({EENHEID_KORT[kenmerk]})"
    as_opties = {"labelExpr": NL_DECIMAAL if kenmerk == "gem_huishoudensgrootte" else NL_GETAL}
    if kenmerk == "stedelijkheid":
        as_opties["tickMinStep"] = 1  # alleen hele klassen 1-5
    lijn = alt.Chart(df).mark_line(color=PETROL, strokeWidth=2.5, point=kenmerk == "stedelijkheid").encode(
        x=alt.X("waarde:Q", title=titel, axis=alt.Axis(**as_opties)),
        # Vaste schaal (minstens P10-P90 van alle buurten): zo lijkt een klein effect niet groot
        y=alt.Y("schatting:Q", title="Schatting (kWh)",
                scale=alt.Scale(domain=[min(REF["p10"], df["schatting"].min()) - 100,
                                        max(REF["p90"], df["schatting"].max()) + 100], nice=False),
                axis=alt.Axis(labelExpr=NL_GETAL)),
        tooltip=[alt.Tooltip("waarde_t:N", title=KORTE_LABELS[kenmerk]),
                 alt.Tooltip("schatting_t:N", title="Schatting")])
    lagen = [lijn]
    if invoer.get(kenmerk) is not None:
        lagen.append(alt.Chart(pd.DataFrame({"waarde": [float(invoer[kenmerk])], "schatting": [huidige_schatting]}))
                     .mark_point(filled=True, size=160, color=AMBER, stroke="white", strokeWidth=1.5)
                     .encode(x="waarde:Q", y="schatting:Q"))
    return stijl(alt.layer(*lagen).properties(height=220))


def waarde_tekst(k, v):
    """Kenmerkwaarde zoals een mens hem leest (Nederlandse notatie)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "onbekend"
    if k == "stedelijkheid":
        return STEDELIJKHEID_KORT[int(v)]
    if k == "woz_waarde_x1000":
        return f"€ {aantal(round(v * 1000))}"
    if k == "gem_huishoudensgrootte":
        return nl_getal(v)
    return f"{nl_getal(round(v))}%"


def uitleg_stappen(uitleg, invoer):
    """De stappen van de watervalgrafiek: van de gemiddelde schatting, via elk kenmerk, naar deze schatting."""
    volgorde = sorted(kenmerken, key=lambda k: -abs(uitleg["bijdragen"][k]))
    stappen, stand = [], uitleg["basis"]
    for k in volgorde:
        b = uitleg["bijdragen"][k]
        stappen.append({"stap": KORTE_LABELS[k], "waarde": waarde_tekst(k, invoer.get(k)), "van": stand,
                        "tot": stand + b, "richting": "verhoogt" if b >= 0 else "verlaagt",
                        "bijdrage": "≈ 0 kWh" if abs(b) < 5 else f"{'+' if b >= 0 else '−'}{kwh(abs(b))}"})
        stand += b
    return pd.DataFrame(stappen), stand


def uitleg_grafiek(uitleg, invoer):
    """Watervalgrafiek (SHAP). Korte labels, zodat hij ook op een telefoon past; de getallen staan in de tabel eronder."""
    staven, eind_waarde = uitleg_stappen(uitleg, invoer)
    start, eind = "Gemiddelde schatting", "Deze schatting"
    punten = pd.DataFrame([{"stap": start, "waarde": uitleg["basis"], "kleur": REFERENTIE, "t": kwh(uitleg["basis"])},
                           {"stap": eind, "waarde": eind_waarde, "kleur": AMBER, "t": kwh(eind_waarde)}])
    alles = list(staven["van"]) + list(staven["tot"])
    marge = max(60.0, 0.08 * (max(alles) - min(alles)))
    schaal = alt.Scale(domain=[min(alles) - marge, max(alles) + marge], nice=False)
    y = alt.Y("stap:N", sort=[start] + list(staven["stap"]) + [eind], title=None, axis=alt.Axis(labelLimit=160))
    balken = alt.Chart(staven).mark_bar(height=16, cornerRadius=2).encode(
        x=alt.X("van:Q", scale=schaal, title="kWh per woning",
                  axis=alt.Axis(labelExpr=NL_GETAL, tickCount=4, labelFlush=True)),
        x2="tot:Q", y=y,
        color=alt.Color("richting:N", scale=alt.Scale(domain=["verhoogt", "verlaagt"], range=[PETROL, REFERENTIE]),
                        legend=None),
        tooltip=[alt.Tooltip("stap:N", title="Kenmerk"), alt.Tooltip("waarde:N", title="Waarde"),
                 alt.Tooltip("bijdrage:N", title="Bijdrage")])
    stippen = alt.Chart(punten).mark_point(filled=True, size=150, opacity=1).encode(
        x=alt.X("waarde:Q", scale=schaal), y=y, color=alt.Color("kleur:N", scale=None),
        tooltip=[alt.Tooltip("stap:N", title=" "), alt.Tooltip("t:N", title="Waarde")])
    return stijl(alt.layer(balken, stippen).properties(height=34 * (len(staven) + 2)))


def uitleg_tonen(invoer):
    """Inhoud van de uitklapper 'Waarom deze schatting?': grafiek plus dezelfde getallen als tabel."""
    uitleg = verklaar(invoer, pipeline, metadata)
    if uitleg is None:
        return
    st.altair_chart(uitleg_grafiek(uitleg, invoer), width="stretch")
    staven, eind_waarde = uitleg_stappen(uitleg, invoer)
    tabel_uitleg = pd.concat([
        pd.DataFrame([{"Stap": "Gemiddelde schatting", "Waarde": kwh(uitleg["basis"]), "Bijdrage": ""}]),
        staven.rename(columns={"stap": "Stap", "waarde": "Waarde", "bijdrage": "Bijdrage"})[["Stap", "Waarde", "Bijdrage"]],
        pd.DataFrame([{"Stap": "Deze schatting", "Waarde": kwh(eind_waarde), "Bijdrage": ""}])], ignore_index=True)
    st.dataframe(tabel_uitleg, hide_index=True, width="stretch")
    tekst = (f"Het model begint bij de gemiddelde schatting over de trainingsbuurten ({kwh(uitleg['basis'])}). "
             "Elke balk laat zien hoeveel één kenmerk de schatting omhoog (+, petrol) of omlaag (−, grijsblauw) duwt; "
             "samen komen ze uit op deze schatting. Berekend met SHAP. Dit verklaart het **model**, "
             "geen oorzaak in de werkelijkheid.")
    if any(invoer.get(k) is None for k in kenmerken):
        tekst += (" Bij een kenmerk dat 'onbekend' is, laat de balk het effect zien van het ontbreken zelf "
                  "(de ingevulde mediaan plus, waar het model die heeft, de markering 'ontbrak').")
    toelichting(tekst)


def belang_grafiek(belang):
    """Hoeveel stijgt de test-MAE als je één kenmerk door elkaar husselt?"""
    df = pd.DataFrame([{"kenmerk": KORTE_LABELS[k], "toename": v["toename_mae_gemiddeld"],
                        "laag": v["toename_mae_gemiddeld"] - v["toename_mae_sd"],
                        "hoog": v["toename_mae_gemiddeld"] + v["toename_mae_sd"],
                        "tekst": f"+{aantal(round(v['toename_mae_gemiddeld']))} kWh",
                        "sd": f"± {nl_getal(round(v['toename_mae_sd'], 1))} kWh"}
                       for k, v in belang["kenmerken"].items()])
    volgorde = df.sort_values("toename", ascending=False)["kenmerk"].tolist()
    y = alt.Y("kenmerk:N", sort=volgorde, title=None, axis=alt.Axis(labelLimit=200))
    x_titel = "Toename test-MAE (kWh)"
    balken = alt.Chart(df).mark_bar(color=PETROL, height=16, cornerRadiusEnd=3).encode(
        x=alt.X("toename:Q", title=x_titel, axis=alt.Axis(labelExpr=NL_GETAL, tickCount=4, labelFlush=True),
                scale=alt.Scale(domain=[0, float(df["hoog"].max()) * 1.3], nice=False)), y=y,
        tooltip=[alt.Tooltip("kenmerk:N", title="Kenmerk"), alt.Tooltip("tekst:N", title="Toename MAE"),
                 alt.Tooltip("sd:N", title="Spreiding (10 keer)")])
    getallen = alt.Chart(df).mark_text(align="left", dx=6, fontSize=11, color="#1C2430").encode(
        x="toename:Q", y=y, text="tekst:N")
    return stijl((balken + getallen).properties(
        height=220, autosize=alt.AutoSizeParams(type="fit", contains="padding")))


def eigen_verbruik_veld(key):
    return st.number_input("Jouw eigen levering uit het net (kWh per jaar, optioneel)", min_value=100.0,
                           max_value=25000.0, value=None, step=100.0, format="%.0f", key=key,
                           placeholder="bijv. 2800 (zonder punt)",
                           help="Staat op je jaarafrekening als 'levering' (normaal + dal samen). Typ het getal "
                                "zonder punt. Heb je zonnepanelen, dan is de vergelijking ruw: de CBS-bron beschrijft "
                                "niet precies hoe teruglevering is verwerkt. Je getal wordt niet bewaard.")


def eigen_verbruik_tekst(eigen, vergelijk, naam):
    verschil = eigen - vergelijk
    richting = "hoger" if verschil >= 0 else "lager"
    toelichting(f"Jouw levering ({aantal(eigen)} kWh) is **{kwh(abs(verschil))} {richting}** dan {naam}. Let op: dat is "
                "één huishouden tegenover een gemiddelde over alle woningen in een buurt. Een groot verschil is "
                "normaal (gezinsgrootte, woninggrootte, zonnepanelen, elektrisch koken of rijden) en zegt niet of "
                "je 'te veel' gebruikt.")


# ----------------------------------------------------------------------
# Kop
# ----------------------------------------------------------------------
st.title("Elektriciteitslevering per woning")
with st.container(horizontal=True, gap="small"):
    st.badge(f"kWh per woning per jaar · {doel['referentiejaar']} · CBS-buurt", color="gray")
    st.badge(f"Getest op {test['n_gemeenten']} ongeziene gemeenten", color="green")
toelichting("Schatting van de gemiddelde jaarlijkse levering **uit het net** per particuliere woning, "
            "op basis van buurtkenmerken.")

tab_schatting, tab_buurt, tab_gemeente, tab_model, tab_over = st.tabs(
    ["Schatting", "Buurt", "Gemeenten",
     "Model", "Uitleg"])


# ======================================================================
# Tab 1: schatting maken
# ======================================================================
def voorbeelden():
    """Typische buurten: mediaan van de trainingsbuurten per stedelijkheidsklasse (echte data)."""
    uit = {}
    for klasse in range(1, 6):
        groep = train[train["stedelijkheid"] == klasse]
        waarden = {k: float(groep[k].median()) for k in kenmerken}
        waarden["stedelijkheid"] = klasse
        uit[f"Typische {STEDELIJKHEID_LANG[klasse].split(' (')[0]} buurt"] = waarden
    return uit


VOORBEELDEN = voorbeelden()
for k in kenmerken:  # beginwaarden: leeg
    st.session_state.setdefault(f"in_{k}", None)


def afronden(k, waarde):
    if k == "gem_huishoudensgrootte":
        return round(waarde, 1)
    if k == "stedelijkheid":
        return int(waarde)
    return float(round(waarde))


def voorbeeld_invullen():
    """Callback van de voorbeeldkeuze: vult de invoervelden en zet de keuze terug."""
    keuze = st.session_state.get("voorbeeld")
    if keuze == "Alles leegmaken":
        for k in kenmerken:
            st.session_state[f"in_{k}"] = None
    elif keuze in VOORBEELDEN:
        for k, w in VOORBEELDEN[keuze].items():
            st.session_state[f"in_{k}"] = afronden(k, w)
    st.session_state.pop("resultaat", None)  # oud resultaat hoort niet meer bij de nieuwe invoer
    st.session_state["voorbeeld"] = None


def invoerveld(k):
    spec = INVOER_SPEC[k]
    b = bereik[k]
    if k == "stedelijkheid":
        st.selectbox(f"{KORTE_LABELS[k]} (CBS-klasse)", options=[None] + list(STEDELIJKHEID_LANG),
                     format_func=lambda v: "onbekend" if v is None else f"{v} · {STEDELIJKHEID_LANG[v]}",
                     key=f"in_{k}", placeholder="onbekend",
                     help="Gebaseerd op het aantal adressen per km² rond de buurt.")
    else:
        st.number_input(f"{KORTE_LABELS[k]} ({EENHEID_KORT[k]})", min_value=float(spec["min"]),
                        max_value=float(spec["max"]), step=0.1 if k == "gem_huishoudensgrootte" else 1.0,
                        format="%.1f" if k == "gem_huishoudensgrootte" else "%.0f",
                        key=f"in_{k}", placeholder="onbekend",
                        help=f"{spec['label']} ({spec['eenheid']}). In de trainingsdata: {nl_getal(b['min'])} tot "
                             f"{nl_getal(b['max'])}, mediaan {nl_getal(b['mediaan'])}. Laat leeg als onbekend.")


with tab_schatting:
    links, rechts = st.columns([1.05, 1], gap="large")

    with links:
        st.subheader("Buurtkenmerken")
        st.selectbox("Begin met een voorbeeld", options=list(VOORBEELDEN) + ["Alles leegmaken"], index=None,
                     key="voorbeeld", on_change=voorbeeld_invullen, placeholder="Kies een typische buurt (optioneel)",
                     help="Vult de mediaanwaarden in van trainingsbuurten met die stedelijkheid. Pas daarna gerust aan.")
        with st.form("invoer", border=True):
            st.markdown("**Woningen**")
            k1, k2 = st.columns(2)
            for i, k in enumerate(WONING_VELDEN):
                with (k1 if i % 2 == 0 else k2):
                    invoerveld(k)
            st.markdown("**Huishoudens en omgeving**")
            k1, k2 = st.columns(2)
            with k1:
                invoerveld("gem_huishoudensgrootte")
            with k2:
                invoerveld("stedelijkheid")
            verzonden = st.form_submit_button("Schatting berekenen", type="primary",
                                              width="stretch")
        toelichting("Percentages als 0–100. Lege velden mogen: het model behandelt ze als onbekend, "
                    "maar de schatting wordt minder betrouwbaar.")

    if verzonden:
        invoer = {k: st.session_state[f"in_{k}"] for k in kenmerken}
        if all(v is None for v in invoer.values()):
            st.session_state["resultaat"] = {"invoer": invoer, "voorspelling_kwh": None, "waarschuwingen": [],
                                             "fouten": ["Vul minstens één kenmerk in, of kies een voorbeeld."]}
        else:
            st.session_state["resultaat"] = {"invoer": invoer, **voorspel(invoer, pipeline, metadata)}

    with rechts:
        st.subheader("Resultaat")
        resultaat = st.session_state.get("resultaat")
        with st.container(border=True):
            if resultaat is None:
                # Lege status
                st.markdown("##### Nog geen schatting")
                st.write("Vul de buurtkenmerken in of kies een **voorbeeld**, en klik op **Schatting berekenen**.")
                st.altair_chart(leveringsmeter([]), width="stretch")
                toelichting(BANDUITLEG + " Je schatting verschijnt als amberkleurige stip.")
            elif resultaat["fouten"]:
                # Foutstatus
                st.markdown("##### Kan geen schatting maken")
                for fout in resultaat["fouten"]:
                    st.error(fout)
                st.write("Pas de invoer aan en probeer het opnieuw.")
            else:
                # Succes (eventueel met gedeeltelijke invoer)
                waarde = resultaat["voorspelling_kwh"]
                if any(st.session_state.get(f"in_{k}") != v for k, v in resultaat["invoer"].items()):
                    st.info("De invoer is gewijzigd. Klik opnieuw op **Schatting berekenen** om bij te werken.")
                st.metric("Geschatte levering uit het net", kwh(waarde))
                toelichting(f"gemiddeld per particuliere woning per jaar · {doel['referentiejaar']} · "
                            f"model {metadata['model_versie'].rsplit('-', 1)[-1]}")
                meter_plek = st.container()  # de meter wordt pas getekend als het eigen verbruik bekend is
                pct = percentiel(waarde, train_doel)
                st.write(f"Hoger dan **{nl_getal(round(pct))}%** van de {aantal(REF['n'])} trainingsbuurten "
                         f"(mediaan {kwh(REF['p50'])}).")
                eigen = eigen_verbruik_veld("eigen_kwh_schatting")
                markeringen = [{"waarde": waarde, "label": "jouw schatting", "kleur": AMBER, "vorm": "circle"}]
                if eigen is not None:
                    markeringen.append({"waarde": eigen, "label": "jouw levering", "kleur": EIGEN, "vorm": "square"})
                with meter_plek:
                    st.altair_chart(leveringsmeter(markeringen), width="stretch")
                    toelichting(BANDUITLEG + (" Amber stip = schatting, donker vierkant = jouw levering."
                                              if eigen is not None else ""))
                if eigen is not None:
                    eigen_verbruik_tekst(eigen, waarde, "deze schatting")
                st.info(f"**Hoe nauwkeurig?** In {test['n_gemeenten']} gemeenten die het model niet kende, zat de "
                        f"schatting gemiddeld **{kwh(test['MAE'])}** naast de CBS-waarde. Dat is een gemiddelde, "
                        f"geen garantie voor één buurt.")
                onbekend = [KORTE_LABELS[k] for k, v in resultaat["invoer"].items() if v is None]
                if onbekend:
                    st.warning(f"**Gedeeltelijke invoer.** Onbekend: {', '.join(onbekend)}. Het model behandelt "
                               "deze als ontbrekend, net als bij buurten waarvoor het CBS geen cijfer publiceert. "
                               "In de test was de fout bij buurten met ontbrekende kenmerken ruim twee keer zo groot; "
                               "voor deze precieze combinatie van lege velden is de fout niet apart getest.")
                for w in resultaat["waarschuwingen"]:
                    if "niet ingevuld" not in w:
                        st.warning(w)
                with st.expander("Wat als één kenmerk anders is?"):
                    keuze = st.selectbox("Kenmerk", kenmerken, format_func=lambda k: KORTE_LABELS[k],
                                         key="wat_als_kenmerk")
                    st.altair_chart(wat_als_grafiek(resultaat["invoer"], keuze, waarde), width="stretch")
                    toelichting("De lijn laat zien wat het **model** voorspelt als alleen dit kenmerk verandert. "
                                "Dat is geen oorzaak: in echte buurten veranderen kenmerken samen.")
                    if resultaat["invoer"].get(keuze) is None:
                        toelichting("Je liet dit kenmerk leeg. De lijn gaat ervan uit dat het wél bekend is; "
                                    "jouw schatting ligt daarom niet op de lijn.")
                with st.expander("Waarom deze schatting?"):
                    uitleg_tonen(resultaat["invoer"])
                export = pd.DataFrame(
                    [{"onderdeel": INVOER_SPEC[k]["label"], "waarde": waarde_tekst(k, v)}
                     for k, v in resultaat["invoer"].items()]
                    + [{"onderdeel": "Geschatte levering uit het net (kWh per woning per jaar)", "waarde": kwh(waarde, False)},
                       {"onderdeel": "Percentage trainingsbuurten met een lagere levering (%)", "waarde": nl_getal(round(pct))},
                       {"onderdeel": "Test-MAE van het model, alle testbuurten (kWh)", "waarde": kwh(test["MAE"], False)}]
                    + ([{"onderdeel": "Test-MAE bij testbuurten met een ontbrekend kenmerk (kWh)",
                         "waarde": kwh(metrics["fouten_per_subgroep_test"]["kenmerk_ontbreekt"]["True"]["MAE"], False)},
                        {"onderdeel": "Waarschuwing", "waarde": "Gedeeltelijke invoer. Onbekend: " + ", ".join(onbekend)}]
                       if onbekend else [])
                    + [
                       {"onderdeel": "Model", "waarde": metadata["model_versie"]}]
                    + [{"onderdeel": "Waarschuwing", "waarde": w} for w in resultaat["waarschuwingen"]
                       if "niet ingevuld" not in w])
                st.download_button("Schatting downloaden (CSV)", data=csv_bytes(export), file_name="schatting.csv",
                                   mime="text/csv", key="download_schatting")

    # ------------------------------------------------------------------
    # Meerdere buurten tegelijk (zelfde route als src/predict.py --input)
    # ------------------------------------------------------------------
    with st.expander("Meerdere buurten tegelijk schatten (CSV-bestand)"):
        st.write("Upload een CSV met één rij per buurt en deze kolommen: "
                 + ", ".join(f"`{k}`" for k in kenmerken) + ". Lege cellen mogen (onbekend). "
                 f"Andere kolommen, zoals een buurtnaam, blijven staan in de uitvoer. Maximaal {aantal(MAX_BATCH_RIJEN)} rijen.")
        sjabloon = pd.DataFrame([{"naam": naam, **{k: afronden(k, w) for k, w in waarden.items()}}
                                 for naam, waarden in VOORBEELDEN.items()], columns=["naam"] + kenmerken)
        st.download_button("Voorbeeldbestand downloaden", data=csv_bytes(sjabloon), file_name="voorbeeld_buurten.csv",
                           mime="text/csv", key="download_sjabloon")
        bestand = st.file_uploader("CSV-bestand: sleep het in het vak of klik op de knop om te bladeren",
                                   type=["csv"], key="batch_bestand")
        if bestand is not None:
            invoer_tabel, leesfout = None, None
            # Eerst UTF-8 (ook met BOM, zoals Excel 'CSV UTF-8'), anders Windows-1252 (Excel 'CSV' in het Nederlands)
            for codering in ["utf-8-sig", "cp1252"]:
                try:
                    bestand.seek(0)
                    # sep=None: herkent zelf komma of puntkomma; als tekst inlezen zodat '2,4' ook werkt
                    invoer_tabel = pd.read_csv(bestand, sep=None, engine="python", dtype=str, encoding=codering)
                    break
                except UnicodeDecodeError as fout:
                    leesfout = fout
                except Exception as fout:  # onleesbaar bestand: melding in plaats van een traceback
                    leesfout = fout
                    break
            if invoer_tabel is None:
                st.error(f"Dit bestand kan niet als CSV worden gelezen ({type(leesfout).__name__}). "
                         "Gebruik het voorbeeldbestand als begin.")
            if invoer_tabel is not None:
                invoer_tabel.columns = [str(c).replace("\ufeff", "").strip() for c in invoer_tabel.columns]
                ontbrekend = [k for k in kenmerken if k not in invoer_tabel.columns]
                if ontbrekend:
                    st.error("Deze kolommen ontbreken: " + ", ".join(f"`{k}`" for k in ontbrekend)
                             + ". Kolomnamen moeten precies zo heten als in het voorbeeldbestand.")
                elif len(invoer_tabel) == 0:
                    st.warning("Het bestand bevat geen rijen.")
                elif len(invoer_tabel) > MAX_BATCH_RIJEN:
                    st.error(f"Het bestand heeft {aantal(len(invoer_tabel))} rijen; het maximum is "
                             f"{aantal(MAX_BATCH_RIJEN)}. Gebruik voor grote bestanden `python src/predict.py`.")
                else:
                    uitkomst_batch = voorspel_batch(invoer_tabel, pipeline, metadata)
                    gelukt = int(uitkomst_batch["voorspelling_kwh"].notna().sum())
                    met_fout = int((uitkomst_batch["fouten"] != "").sum())
                    met_waarschuwing = int(((uitkomst_batch["waarschuwingen"] != "")
                                            & uitkomst_batch["voorspelling_kwh"].notna()).sum())
                    st.write(f"**{aantal(gelukt)} van {aantal(len(uitkomst_batch))}** rijen geschat · "
                             f"{aantal(met_fout)} met een fout (geen schatting) · {aantal(met_waarschuwing)} met een "
                             "waarschuwing.")
                    uitkomst_batch["voorspelling_kwh"] = uitkomst_batch["voorspelling_kwh"].round(-1)
                    weergave = uitkomst_batch.head(200).fillna("").copy()
                    weergave["voorspelling_kwh"] = uitkomst_batch["voorspelling_kwh"].head(200).map(
                        lambda v: "geen schatting" if pd.isna(v) else kwh(v))
                    st.dataframe(weergave.rename(columns={"voorspelling_kwh": "schatting"}), hide_index=True,
                                 width="stretch", height=260)
                    if len(uitkomst_batch) > 200:
                        toelichting("Hierboven staan de eerste 200 rijen; de download bevat alles.")
                    st.download_button("Resultaat downloaden (CSV)", data=csv_bytes(uitkomst_batch),
                                       file_name="voorspellingen.csv", mime="text/csv", key="download_batch")
        toelichting("Downloads gebruiken een puntkomma en een decimale komma, zodat Excel in het Nederlands ze "
                    "goed opent. Een geüpload bestand wordt alleen gebruikt voor deze tabel en niet bewaard.")


# ======================================================================
# Tab 2: bestaande buurt
# ======================================================================
BUURT_INDEX = tabel.set_index("buurt_code")
GEMEENTENAMEN = sorted(tabel["gemeente_naam"].unique())


def postcode_zoeken():
    """Callback van 'Zoeken': zet gemeente en buurt in de keuzelijsten (vóór ze getekend worden)."""
    st.session_state.pop("postcode_melding", None)
    try:
        schoon, gevonden = zoek_buurten(st.session_state.get("postcode", ""))
    except PostcodeFout as fout:
        st.session_state["postcode_melding"] = ("error", str(fout), None)
        return
    pc = f"{schoon[:4]} {schoon[4:]}"
    bekend = [g for g in gevonden if g["buurt_code"] in BUURT_INDEX.index]
    if bekend:
        eerste = bekend[0]
        st.session_state["buurt_gemeente"] = BUURT_INDEX.loc[eerste["buurt_code"], "gemeente_naam"]
        st.session_state["vergelijk_gemeente"] = st.session_state["buurt_gemeente"]
        st.session_state["buurt_code"] = eerste["buurt_code"]
        tekst = f"Postcode {pc} ligt in buurt **{BUURT_INDEX.loc[eerste['buurt_code'], 'buurt_naam'].rstrip('.')}**."
        andere = [f"{BUURT_INDEX.loc[g['buurt_code'], 'buurt_naam']} ({g['adressen']} adressen)" for g in bekend[1:]]
        if andere:
            tekst += (f" De meeste adressen ({eerste['adressen']}) liggen daar; de postcode valt ook in: "
                      + ", ".join(andere) + ". Kies die hieronder als je adres daar ligt.")
        st.session_state["postcode_melding"] = ("success", tekst, eerste["buurt_code"])
    else:
        eerste = gevonden[0]
        tekst = (f"Postcode {pc} ligt volgens PDOK in buurt {eerste['buurt_naam']} ({eerste['buurt_code']}), maar die "
                 f"code bestaat niet in de CBS-indeling van {doel['referentiejaar']}: buurtgrenzen en -codes veranderen "
                 "bijna elk jaar.")
        if eerste["gemeente_naam"] in GEMEENTENAMEN:
            eerste_buurt = (tabel[tabel["gemeente_naam"] == eerste["gemeente_naam"]]
                            .sort_values("buurt_naam")["buurt_code"].iloc[0])
            st.session_state["buurt_gemeente"] = eerste["gemeente_naam"]
            st.session_state["vergelijk_gemeente"] = eerste["gemeente_naam"]
            st.session_state["buurt_code"] = eerste_buurt
            tekst += (f" Hieronder staat nu de eerste buurt van {eerste['gemeente_naam']} in de lijst; dat is "
                      "waarschijnlijk niet jouw buurt. Kies je buurt zelf.")
            st.session_state["postcode_melding"] = ("warning", tekst, eerste_buurt)
        else:
            tekst += f" Ook gemeente {eerste['gemeente_naam']} staat niet in de lijst van 2023. Kies zelf een buurt."
            st.session_state["postcode_melding"] = ("warning", tekst, st.session_state.get("buurt_code"))


with tab_buurt:
    st.subheader("Een echte buurt controleren")
    with st.form("postcode_formulier", border=False, enter_to_submit=True):
        p1, p2 = st.columns([2.2, 1], vertical_alignment="bottom")
        p1.text_input("Zoek op postcode", key="postcode", placeholder="bijv. 5611 AA", max_chars=8)
        p2.form_submit_button("Zoeken", on_click=postcode_zoeken, width="stretch")
    melding = st.session_state.get("postcode_melding")
    # Een zoekresultaat tonen zolang de gevonden buurt nog gekozen is (of bij een fout direct na het zoeken)
    if melding and (melding[0] == "error" or melding[2] == st.session_state.get("buurt_code")):
        getattr(st, melding[0])(melding[1])
    toelichting("Je postcode gaat naar de PDOK Locatieserver van de overheid om de buurt op te zoeken; "
                "de app bewaart hem niet. Je kunt ook hieronder zelf kiezen.")

    k1, k2 = st.columns([1, 1.6])
    gemeente = k1.selectbox("Gemeente", GEMEENTENAMEN, index=GEMEENTENAMEN.index("Eindhoven"), key="buurt_gemeente",
                            on_change=lambda: st.session_state.update(vergelijk_gemeente=st.session_state["buurt_gemeente"],
                                                                      postcode_melding=None))
    buurten = tabel[tabel["gemeente_naam"] == gemeente].sort_values("buurt_naam").set_index("buurt_code")
    buurt_code = k2.selectbox("Buurt", buurten.index,
                              format_func=lambda c: f"{buurten.loc[c, 'buurt_naam']} ({c})", key="buurt_code")
    rij = buurten.loc[buurt_code]
    invoer_buurt = {k: (None if pd.isna(rij[k]) else float(rij[k])) for k in kenmerken}
    gemeten = rij[DOELKOLOM]
    # Zonder woningen bestaat er geen "gemiddelde per woning": dan geen schatting tonen
    geen_woningen = pd.isna(gemeten) and rij["woningvoorraad"] == 0
    uitkomst = ({"voorspelling_kwh": None, "fouten": [], "waarschuwingen": []} if geen_woningen
                else voorspel(invoer_buurt, pipeline, metadata))
    schatting = uitkomst["voorspelling_kwh"]

    with st.container(border=True):
        status = {"train": ("In de trainingsset", "orange"), "test": ("In de testset", "green")}.get(
            rij["set"], ("Niet gebruikt: geen CBS-waarde", "gray"))
        st.badge(status[0], color=status[1])
        m1, m2, m3 = st.columns(3)
        m1.metric("CBS-waarde 2023", "niet gepubliceerd" if pd.isna(gemeten) else kwh(gemeten))
        m2.metric("Schatting model", "geen woningen" if geen_woningen else ("–" if schatting is None else kwh(schatting)))
        if not pd.isna(gemeten) and schatting is not None:
            verschil = schatting - gemeten
            m3.metric("Verschil (schatting − CBS)", f"{'+' if verschil >= 0 else '−'}{kwh(abs(verschil))}")
        else:
            m3.metric("Verschil", "–")

        markeringen = []
        if not pd.isna(gemeten):
            markeringen.append({"waarde": float(gemeten), "label": "CBS", "kleur": PETROL, "vorm": "diamond"})
        if schatting is not None:
            markeringen.append({"waarde": schatting, "label": "schatting", "kleur": AMBER, "vorm": "circle"})
        meter_plek = st.container()
        eigen = eigen_verbruik_veld("eigen_kwh_buurt") if markeringen else None
        if eigen is not None:
            markeringen.append({"waarde": eigen, "label": "jouw levering", "kleur": EIGEN, "vorm": "square"})
        if markeringen:
            with meter_plek:
                st.altair_chart(leveringsmeter(markeringen), width="stretch")
                uitleg_vormen = [v for v, aan in [("open ruit = CBS-waarde", not pd.isna(gemeten)),
                                                  ("gevulde stip = schatting", schatting is not None),
                                                  ("donker vierkant = jouw levering", eigen is not None)] if aan]
                vormen = ", ".join(uitleg_vormen)
                toelichting(f"{BANDUITLEG} {vormen[:1].upper()}{vormen[1:]}.")
        if eigen is not None:
            if not pd.isna(gemeten):
                eigen_verbruik_tekst(eigen, float(gemeten), "het CBS-gemiddelde van deze buurt")
            elif schatting is not None:
                eigen_verbruik_tekst(eigen, schatting, "de schatting voor deze buurt")

        if rij["set"] == "train":
            toelichting("Deze buurt zat in de trainingsdata. Een goede schatting hier zegt weinig over nieuwe "
                        "buurten; kijk daarvoor naar buurten uit de testset of naar de testfout.")
        elif rij["set"] == "test":
            toelichting("Het model heeft deze gemeente tijdens het trainen niet gezien: dit is een eerlijke test.")
        elif geen_woningen:
            toelichting("Deze buurt heeft geen woningen (bijvoorbeeld een bedrijventerrein of natuurgebied). "
                        "Een gemiddelde per woning bestaat hier niet, dus het model maakt geen schatting.")
        else:
            toelichting("Het CBS heeft voor deze buurt geen cijfer gepubliceerd (meestal omdat er minder dan "
                        "6 bewoonde woningen zijn, soms vanwege geheimhouding). Deze buurt deed niet mee aan "
                        "training of test.")
        ontbreekt = [KORTE_LABELS[k] for k, v in invoer_buurt.items() if v is None]
        if ontbreekt and not geen_woningen:
            st.warning(f"**Gedeeltelijke gegevens.** Het CBS publiceert voor deze buurt niet: {', '.join(ontbreekt)}. "
                       "Het model behandelt deze als onbekend; de schatting is daardoor minder betrouwbaar.")
        for w in uitkomst["waarschuwingen"]:
            if "niet ingevuld" not in w:
                st.warning(w)
        if schatting is not None:
            with st.expander("Waarom deze schatting?"):
                uitleg_tonen(invoer_buurt)

    # ------------------------------------------------------------------
    # Vergelijken: deze buurt, haar gemeente, Nederland en eventueel een tweede buurt
    # ------------------------------------------------------------------
    st.markdown("**Vergelijken**")
    v1, v2 = st.columns([1, 1.6])
    gemeente_2 = v1.selectbox("Vergelijk met een buurt in", GEMEENTENAMEN, index=GEMEENTENAMEN.index(gemeente),
                              key="vergelijk_gemeente")
    buurten_2 = tabel[tabel["gemeente_naam"] == gemeente_2].sort_values("buurt_naam").set_index("buurt_code")
    code_2 = v2.selectbox("Tweede buurt (optioneel)", buurten_2.index, index=None,
                          format_func=lambda c: f"{buurten_2.loc[c, 'buurt_naam']} ({c})", key="vergelijk_buurt",
                          placeholder="Kies een buurt")
    gem_rij = regio[(regio["soort"] == "gemeente") & (regio["gemeente_naam"] == gemeente)]
    nl_rij = regio[regio["soort"] == "land"]
    kolommen = [(f"{rij['buurt_naam']} (buurt)", rij, True)]
    if code_2 is not None and code_2 != buurt_code:
        titel_2 = f"{buurten_2.loc[code_2, 'buurt_naam']} (buurt)"
        if titel_2 == kolommen[0][0]:  # zelfde naam: codes erbij, anders valt er een kolom weg
            kolommen[0] = (f"{rij['buurt_naam']} ({buurt_code})", rij, True)
            titel_2 = f"{buurten_2.loc[code_2, 'buurt_naam']} ({code_2})"
        kolommen.append((titel_2, buurten_2.loc[code_2], True))
    if len(gem_rij):
        kolommen.append((f"{gemeente} (gemeente)", gem_rij.iloc[0], False))
    if len(nl_rij):
        kolommen.append(("Nederland", nl_rij.iloc[0], False))

    def schatting_tekst(r, is_buurt):
        if not is_buurt:
            return "– (alleen voor buurten)"
        if pd.isna(r[DOELKOLOM]) and r["woningvoorraad"] == 0:
            return "geen woningen"
        u = voorspel({k: (None if pd.isna(r[k]) else float(r[k])) for k in kenmerken}, pipeline, metadata)
        return "–" if u["voorspelling_kwh"] is None else kwh(u["voorspelling_kwh"])

    vergelijking = {"Kenmerk": ["Levering per woning (CBS)", "Schatting model"]
                    + [f"{KORTE_LABELS[k]} ({EENHEID_KORT[k]})" for k in kenmerken] + ["Woningen"]}
    for titel, r, is_buurt in kolommen:
        vergelijking[titel] = (["niet gepubliceerd" if pd.isna(r[DOELKOLOM]) else kwh(r[DOELKOLOM]),
                                schatting_tekst(r, is_buurt)]
                               + ["onbekend" if pd.isna(r[k]) else (STEDELIJKHEID_KORT[int(r[k])] if k == "stedelijkheid"
                                                                     else nl_getal(r[k])) for k in kenmerken]
                               + [aantal(r["woningvoorraad"])])
    st.dataframe(pd.DataFrame(vergelijking), hide_index=True, width="stretch")
    toelichting("Gemeente- en Nederlandcijfers zijn door het CBS berekend over alle woningen daar (CBS 2023). "
                "Het model maakt alleen schattingen voor buurten. Verschillen in kenmerken laten zien waar buurten "
                "op elkaar lijken of niet; ze bewijzen niet wát het verschil in levering veroorzaakt.")


# ======================================================================
# Tab 3: gemeenten vergelijken
# ======================================================================
with tab_gemeente:
    st.subheader("Gemeenten vergelijken")
    met_cijfer = (gemeenten.dropna(subset=[DOELKOLOM])
                  .sort_values([DOELKOLOM, "gemeente_naam"], kind="stable").reset_index(drop=True))
    # CBS rondt af op tientallen; gemeenten met dezelfde waarde krijgen dezelfde (gedeelde) plek
    met_cijfer["rang"] = met_cijfer[DOELKOLOM].rank(method="min").astype(int)
    namen = met_cijfer["gemeente_naam"].tolist()
    gekozen = st.selectbox("Gemeente", namen, index=namen.index("Eindhoven"), key="gemeente_keuze")
    g = met_cijfer[met_cijfer["gemeente_naam"] == gekozen].iloc[0]
    buurten_g = tabel[(tabel["gemeente_naam"] == gekozen) & tabel[DOELKOLOM].notna()]

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Gemiddelde per woning (CBS)", kwh(g[DOELKOLOM]))
        gedeeld = int((met_cijfer[DOELKOLOM] == g[DOELKOLOM]).sum())
        m2.metric("Plek (laag → hoog)", f"{'gedeeld ' if gedeeld > 1 else ''}{g['rang']} van {len(met_cijfer)}")
        m3.metric("Buurten met cijfer", aantal(len(buurten_g)))
        toelichting("Het gemeentecijfer is door het CBS zelf berekend over alle woningen. "
                    "Het is dus niet het gemiddelde van de buurtcijfers hieronder."
                    + (f" {gedeeld} gemeenten hebben dezelfde afgeronde waarde." if gedeeld > 1 else ""))

        if len(buurten_g):
            punten = buurten_g[["buurt_naam", DOELKOLOM]].rename(columns={DOELKOLOM: "waarde"})
            punten["kwh"] = punten["waarde"].map(kwh)
            laag = min(REF["p01"], punten["waarde"].min()) - 150
            hoog = max(REF["p99"], punten["waarde"].max()) + 150
            schaal = alt.Scale(domain=[laag, hoog], nice=False)
            # Verborgen verticale schaal 0-1 (zoals in de leveringsmeter): band 0,30-0,70, streepjes in het midden
            def y(veld):
                return alt.Y(f"{veld}:Q", scale=alt.Scale(domain=[0, 1]), axis=None)

            band = alt.Chart(pd.DataFrame({"van": [REF["p10"]], "tot": [REF["p90"]], "y1": [0.30], "y2": [0.70]})).mark_rect(
                color=BAND, cornerRadius=3).encode(x=alt.X("van:Q", scale=schaal), x2="tot:Q", y=y("y1"), y2="y2:Q")
            streepjes = alt.Chart(punten.assign(y1=0.25, y2=0.75)).mark_rule(color=PETROL, strokeWidth=1.5, opacity=0.55).encode(
                x=alt.X("waarde:Q", scale=schaal, title="kWh per woning per jaar (per buurt)",
                        axis=alt.Axis(labelExpr=NL_GETAL, tickCount=7, grid=False)),
                y=y("y1"), y2="y2:Q",
                tooltip=[alt.Tooltip("buurt_naam:N", title="Buurt"), alt.Tooltip("kwh:N", title="Waarde")])
            gem_df = pd.DataFrame({"waarde": [float(g[DOELKOLOM])], "y1": [0.12], "y2": [0.78], "yt": [0.93],
                                   "t": [f"{gekozen}: {kwh(g[DOELKOLOM])}"]})
            gem = alt.Chart(gem_df).mark_rule(color=AMBER, strokeWidth=4).encode(
                x=alt.X("waarde:Q", scale=schaal), y=y("y1"), y2="y2:Q",
                tooltip=[alt.Tooltip("t:N", title="Gemeente (CBS)")])
            gem_tekst = alt.Chart(gem_df).mark_text(color=AMBER_TEKST, fontWeight="bold", fontSize=12, align="center").encode(
                x=alt.X("waarde:Q", scale=schaal), y=y("yt"), text="t:N")
            st.altair_chart(stijl((band + streepjes + gem + gem_tekst).properties(height=120)), width="stretch")
            toelichting(f"Elk streepje is een buurt in {gekozen}. Amber lijn = gemeentecijfer. Grijze band = 80% van "
                        "de trainingsbuurten in heel Nederland.")
        else:
            st.info("Voor deze gemeente zijn geen buurtcijfers gepubliceerd.")

    st.markdown("**Alle gemeenten**")
    ranglijst = pd.DataFrame({
        "Plek": met_cijfer["rang"],
        "Gemeente": met_cijfer["gemeente_naam"],
        "kWh per woning": met_cijfer[DOELKOLOM].map(lambda v: kwh(v, False)),   # Nederlandse notatie
        "Verhouding": met_cijfer[DOELKOLOM],                                    # alleen voor de balk
        "Woningen": met_cijfer["woningvoorraad"].map(aantal),
    })
    st.dataframe(ranglijst, hide_index=True, width="stretch", height=320,
                 column_config={"Verhouding": st.column_config.ProgressColumn(
                     " ", format=" ", min_value=0, max_value=float(met_cijfer[DOELKOLOM].max()))})
    st.download_button("Ranglijst downloaden (CSV)", key="download_ranglijst", mime="text/csv",
                       file_name="gemeenten_elektriciteitslevering_2023.csv",
                       data=csv_bytes(pd.DataFrame({
                           "plek": met_cijfer["rang"], "gemeente_code": met_cijfer["gemeente_code"],
                           "gemeente": met_cijfer["gemeente_naam"],
                           "kwh_per_woning_cbs": met_cijfer[DOELKOLOM], "woningen": met_cijfer["woningvoorraad"]})))


@st.cache_data(show_spinner="Schattingen voor alle buurten maken…")
def alle_buurten_csv(vingerafdruk):
    """Schatting voor elke buurt (training, test en zonder CBS-cijfer) als CSV. versie = cache-sleutel."""
    uit = voorspel_batch(tabel[kenmerken], pipeline, metadata)
    geen = tabel[DOELKOLOM].isna() & (tabel["woningvoorraad"] == 0)
    uit.loc[geen.to_numpy(), "voorspelling_kwh"] = np.nan  # geen woningen: geen gemiddelde per woning
    tabel_uit = pd.DataFrame({
        "buurt_code": tabel["buurt_code"], "buurt_naam": tabel["buurt_naam"], "gemeente": tabel["gemeente_naam"],
        "gebruikt_in": tabel["set"].map({"train": "trainingsset", "test": "testset"}).fillna("niet gebruikt (geen CBS-cijfer)"),
        "cbs_kwh": tabel[DOELKOLOM], "schatting_kwh": uit["voorspelling_kwh"].round(-1),
    })
    tabel_uit["verschil_kwh"] = tabel_uit["schatting_kwh"] - tabel_uit["cbs_kwh"]
    tabel_uit = pd.concat([tabel_uit, tabel[kenmerken].reset_index(drop=True),
                           uit[["waarschuwingen"]].rename(columns={"waarschuwingen": "opmerkingen"})], axis=1)
    return csv_bytes(tabel_uit)


# ======================================================================
# Tab 4: modelprestaties
# ======================================================================
with tab_model:
    st.subheader("Hoe goed werkt het model?")
    t, b = metrics["test_gekozen_model"], metrics["test_mediaanbaseline"]
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        # ASCII-min: Streamlit leest het teken van de delta; "inverse" = lager is beter (groen)
        m1.metric("Test-MAE", kwh(t["MAE"]), delta=f"-{kwh(b['MAE'] - t['MAE'])} t.o.v. baseline",
                  delta_color="inverse",
                  help="Gemiddeld absoluut verschil (te hoog of te laag) tussen schatting en CBS-waarde "
                       "in de testgemeenten.")
        m2.metric("Test-RMSE", kwh(t["RMSE"]), help="Zoals MAE, maar grote fouten tellen zwaarder.")
        m3.metric("Test-R²", f"{t['R2']:.2f}".replace(".", ","),
                  help="1 = perfect, 0 = niet beter dan steeds hetzelfde getal. Geen percentage nauwkeurigheid.")
        m4.metric("Testbuurten", aantal(metrics["test_n_buurten"]),
                  help=f"In {metrics['test_n_gemeenten']} gemeenten die niet in de training zaten.")
        toelichting(f"Model: {metrics['gekozen_model'].replace('_', ' ')} · gekozen op de laagste MAE in 5-voudige "
                    f"cross-validatie op de trainingsgemeenten, daarna één keer getest.")

    k1, k2 = st.columns([1.15, 1], gap="large")
    with k1:
        st.markdown("**Werkelijk tegen geschat (testgemeenten)**")
        tp = testvoorspellingen.rename(columns={"werkelijk_kwh": "werkelijk", "voorspeld_kwh": "geschat"})
        tp["cbs_t"], tp["schatting_t"] = tp["werkelijk"].map(kwh), tp["geschat"].map(kwh)
        onder, boven = tp[["werkelijk", "geschat"]].min().min(), tp[["werkelijk", "geschat"]].max().max()
        diagonaal = alt.Chart(pd.DataFrame({"werkelijk": [onder, boven], "geschat": [onder, boven]})).mark_line(
            color=REFERENTIE, strokeDash=[5, 4]).encode(x="werkelijk:Q", y="geschat:Q")
        punten = alt.Chart(tp).mark_circle(size=22, opacity=0.45, color=PETROL).encode(
            x=alt.X("werkelijk:Q", title="CBS-waarde (kWh)", scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=NL_GETAL)),
            y=alt.Y("geschat:Q", title="Schatting (kWh)", scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=NL_GETAL)),
            tooltip=[alt.Tooltip("buurt_naam:N", title="Buurt"), alt.Tooltip("gemeente_naam:N", title="Gemeente"),
                     alt.Tooltip("cbs_t:N", title="CBS"), alt.Tooltip("schatting_t:N", title="Schatting")])
        st.altair_chart(stijl((diagonaal + punten).properties(height=340)), width="stretch")
        # Systematische afwijking aan de randen, berekend uit de testvoorspellingen
        hoog_10 = tp[tp["werkelijk"] >= tp["werkelijk"].quantile(0.9)]
        laag_10 = tp[tp["werkelijk"] <= tp["werkelijk"].quantile(0.1)]
        te_laag = (hoog_10["werkelijk"] - hoog_10["geschat"]).mean()
        te_hoog = (laag_10["geschat"] - laag_10["werkelijk"]).mean()
        toelichting(f"Stippellijn = perfecte schatting. In de 10% testbuurten met de hoogste CBS-waarde lag de "
                    f"schatting gemiddeld **{kwh(te_laag)} te laag**; in de 10% met de laagste waarde gemiddeld "
                    f"**{kwh(te_hoog)} te hoog**. Het model trekt uitschieters dus naar het midden.")
    with k2:
        st.markdown("**Modelvergelijking (cross-validatie)**")
        cv = pd.DataFrame([{"model": n.replace("_", " ") + (" (gekozen)" if n == metrics["gekozen_model"] else ""),
                            "MAE": v["MAE_gemiddeld"], "sd": v["MAE_sd"], "gekozen": n == metrics["gekozen_model"]}
                           for n, v in metrics["cross_validatie"].items()])
        cv["mae_t"], cv["sd_t"] = cv["MAE"].map(kwh), cv["sd"].map(kwh)
        cv["laag"], cv["hoog"] = cv["MAE"] - cv["sd"], cv["MAE"] + cv["sd"]
        volgorde = cv.sort_values("MAE")["model"].tolist()  # beste model bovenaan
        y = alt.Y("model:N", sort=volgorde, title=None, axis=alt.Axis(labelLimit=220))
        balken = alt.Chart(cv).mark_bar(cornerRadiusEnd=3, height=18).encode(
            x=alt.X("MAE:Q", title="Gemiddelde MAE (kWh), lager is beter", axis=alt.Axis(labelExpr=NL_GETAL)), y=y,
            color=alt.condition("datum.gekozen", alt.value(PETROL), alt.value("#AEB8C2")),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("mae_t:N", title="MAE"),
                     alt.Tooltip("sd_t:N", title="Spreiding")])
        foutbalk = alt.Chart(cv).mark_rule(color="#1C2430").encode(
            x=alt.X("laag:Q", title="Gemiddelde MAE (kWh), lager is beter"), x2="hoog:Q", y=y)
        # Getal aan het eind van elke balk: de grafiek is ook zonder kleur te lezen
        getallen = alt.Chart(cv).mark_text(align="left", dx=6, color="#1C2430", fontSize=11).encode(
            x="hoog:Q", y=y, text="mae_t:N")
        st.altair_chart(stijl((balken + foutbalk + getallen).properties(height=230)), width="stretch")

        st.markdown("**Testfout per groep**")
        sub = metrics["fouten_per_subgroep_test"]
        rijen = [{"Groep": f"Stedelijkheid {STEDELIJKHEID_KORT[int(float(k))]}", "Buurten": v["n"],
                  "MAE (kWh)": round(v["MAE"])} for k, v in sub["stedelijkheid"].items() if "MAE" in v]
        rijen += [{"Groep": "Alle kenmerken bekend", "Buurten": sub["kenmerk_ontbreekt"]["False"]["n"],
                   "MAE (kWh)": round(sub["kenmerk_ontbreekt"]["False"]["MAE"])},
                  {"Groep": "Kenmerk ontbreekt", "Buurten": sub["kenmerk_ontbreekt"]["True"]["n"],
                   "MAE (kWh)": round(sub["kenmerk_ontbreekt"]["True"]["MAE"])}]
        groepen = pd.DataFrame(rijen)
        groepen["Buurten"] = groepen["Buurten"].map(aantal)  # Nederlandse notatie (1.188)
        st.dataframe(groepen, hide_index=True, width="stretch",
                     column_config={"MAE (kWh)": st.column_config.ProgressColumn(
                         format="%d", min_value=0, max_value=max(r["MAE (kWh)"] for r in rijen))})

    k1, k2 = st.columns([1.15, 1], gap="large")
    with k1:
        st.markdown("**Op welke kenmerken leunt het model het meest?**")
        if belang:
            st.altair_chart(belang_grafiek(belang), width="stretch")
            spreiding = max(v["toename_mae_sd"] for v in belang["kenmerken"].values())
            toelichting("Per kenmerk: hoeveel groter wordt de testfout (MAE) als we de waarden van dat kenmerk "
                        "willekeurig door elkaar husselen (gemiddelde van 10 keer; de spreiding is hooguit "
                        f"{nl_getal(round(spreiding, 1))} kWh)? Hoe meer, hoe meer het model erop leunt. "
                        "Meergezinswoningen, koopwoningen, huishoudensgrootte en stedelijkheid hangen sterk samen en "
                        "kunnen elkaar deels vervangen; hun belang wordt daardoor te laag ingeschat. Belang is geen oorzaak.")
        else:
            st.info("Nog niet berekend. Voer `python src/explain.py` uit.")
    with k2:
        st.markdown("**Ongewone combinaties**")
        if combinatie:
            c = combinatie
            st.write("Als alle kenmerken bekend zijn, waarschuwt de app bij een combinatie die hoort bij de "
                     f"{nl_getal(c['grens_pct'])}% meest afwijkende combinaties ten opzichte van de trainingsbuurten "
                     "(Local Outlier Factor). Elk kenmerk apart kan dan gewoon zijn.")
            toelichting(combinatie_testuitslag(c).replace(" Wees voorzichtig met deze schatting.", "")
                        + (" Pas vanaf 30 buurten noemen we een foutmaat (dezelfde regel als bij de subgroepen)."
                           if c["test_n_gemarkeerd"] < 30 else ""))
        else:
            st.info("Nog niet berekend. Voer `python src/explain.py` uit.")
        st.markdown("**Gegevens downloaden**")
        st.download_button("Schattingen voor alle buurten (CSV)", key="download_alle_buurten", mime="text/csv",
                           file_name="schattingen_alle_buurten_2023.csv", data=alle_buurten_csv(VINGERAFDRUK))
        toelichting(f"Alle {aantal(len(tabel))} buurten: CBS-waarde (als die is gepubliceerd), schatting (niet bij "
                    "buurten zonder woningen), verschil, kenmerken en of de buurt in de trainingsset of de testset zat. "
                    "Alleen de testbuurten zeggen iets over de nauwkeurigheid.")


# ======================================================================
# Tab 5: uitleg en beperkingen
# ======================================================================
with tab_over:
    delen = model_card.secties(metadata, metrics, config, belang, combinatie, MET_INDICATOR)
    kop1, kop2 = st.columns([3, 1], vertical_alignment="bottom")
    kop1.subheader("Modelkaart")
    kop2.download_button("Modelkaart downloaden (.md)", key="download_modelkaart", mime="text/markdown",
                         file_name="MODEL_CARD.md", width="stretch",
                         data=model_card.markdown(metadata, metrics, config, belang, combinatie, MET_INDICATOR).encode("utf-8"))
    toelichting("Opbouw volgens Mitchell et al. (2019), *Model Cards for Model Reporting*. Alle getallen komen uit "
                "de modelbestanden zelf.")
    k1, k2 = st.columns(2, gap="large")
    for i, (titel, inhoud) in enumerate(delen):
        with (k1 if i < (len(delen) + 1) // 2 else k2):
            st.markdown(f"**{titel}**")
            st.markdown(inhoud)

st.divider()
toelichting(f"DataLab I · groep 4 · model {metadata['model_versie']} · getraind op "
            f"{aantal(metadata['training']['n_buurten'])} buurten in {metadata['training']['n_gemeenten']} gemeenten · "
            f"bron: {doel['bron']}")
