"""
Stap 6: Word-rapport bijwerken met de echte resultaten.

Leest het originele rapport (report/Data Lab-4 (origineel).docx), voegt vóór de
Bronnenlijst een nieuw hoofdstuk toe en zet nieuwe bronnen alfabetisch in de
Bronnenlijst. Alle getallen komen uit reports/*.json en artifacts/*.json, zodat
de tekst altijd klopt met de laatst uitgevoerde analyse.

Schrijft: report/Data Lab-5.docx  (het origineel blijft ongewijzigd)
Uitvoeren: python src/update_report.py
"""

import copy
import json
import os

from docx import Document
from docx.enum.text import WD_BREAK, WD_COLOR_INDEX
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRON = os.path.join(PROJECT_DIR, "report", "Data Lab-4 (origineel).docx")
DOEL = os.path.join(PROJECT_DIR, "report", "Data Lab-5.docx")
FIG = os.path.join(PROJECT_DIR, "reports", "figures")

CBS = "Centraal Bureau voor de Statistiek"
CBS_DATASET = f"({CBS}, 2026e)"


def lees(pad):
    with open(os.path.join(PROJECT_DIR, pad), encoding="utf-8") as f:
        return json.load(f)


def nl(getal, decimalen=0):
    """Getal in Nederlandse notatie: 1.234,5"""
    tekst = f"{getal:,.{decimalen}f}"
    return tekst.replace(",", "X").replace(".", ",").replace("X", ".")


# ----------------------------------------------------------------------
# Hulpfuncties om inhoud toe te voegen
# ----------------------------------------------------------------------
class Schrijver:
    def __init__(self, doc):
        self.doc = doc
        self.tabel_nr = 0
        self.figuur_nr = 0
        # De bestaande tekst gebruikt al "Figuur 1" en "Figuur 2"; wij nummeren door.
        self.figuur_start = 2

    def kop(self, tekst, niveau):
        self.doc.add_heading(tekst, level=niveau)

    def alinea(self, tekst, vet_begin=None):
        p = self.doc.add_paragraph()
        if vet_begin:
            p.add_run(vet_begin).bold = True
        p.add_run(tekst)
        return p

    def opsomming(self, punten):
        for punt in punten:
            p = self.doc.add_paragraph(style="List Paragraph")
            p.add_run("•\t" + punt)

    def aanvullen(self, tekst):
        """Geel gemarkeerde tekst die de groep zelf moet invullen."""
        p = self.doc.add_paragraph()
        r = p.add_run(f"[Aan te vullen door de groep: {tekst}]")
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return p

    def tabel(self, titel, kolommen, rijen, breedtes=None):
        self.tabel_nr += 1
        bijschrift = self.doc.add_paragraph()
        bijschrift.add_run(f"Tabel {self.tabel_nr}. ").bold = True
        bijschrift.add_run(titel)
        t = self.doc.add_table(rows=1, cols=len(kolommen))
        t.style = "Table Grid"
        for i, k in enumerate(kolommen):
            cel = t.rows[0].cells[i]
            cel.text = ""
            cel.paragraphs[0].add_run(str(k)).bold = True
        for rij in rijen:
            cellen = t.add_row().cells
            for i, waarde in enumerate(rij):
                cellen[i].text = str(waarde)
        for rij in t.rows:
            for cel in rij.cells:
                for p in cel.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)
        if breedtes:
            for rij in t.rows:
                for i, b in enumerate(breedtes):
                    rij.cells[i].width = Cm(b)
        self.doc.add_paragraph()

    def figuur(self, bestand, bijschrift, breedte_cm=15):
        self.figuur_nr += 1
        self.doc.add_picture(os.path.join(FIG, bestand), width=Cm(breedte_cm))
        p = self.doc.add_paragraph()
        p.add_run(f"Figuur {self.figuur_start + self.figuur_nr}. ").bold = True
        p.add_run(bijschrift)

    def volgende_figuur(self):
        return self.figuur_start + self.figuur_nr + 1

    def volgende_tabel(self):
        return self.tabel_nr + 1


# ----------------------------------------------------------------------
# Inhoud van het nieuwe hoofdstuk
# ----------------------------------------------------------------------
def testuitslag():
    """Draai de tests en geef de echte uitslag terug (bijv. '64 passed'), zodat het rapport niets aanneemt."""
    import re
    import subprocess
    import sys
    uit = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_DIR, capture_output=True, text=True)
    regel = uit.stdout.strip().splitlines()[-1] if uit.stdout.strip() else ""
    geslaagd = re.search(r"(\d+) passed", regel)
    mislukt = re.search(r"(\d+) failed", regel)
    return int(geslaagd.group(1)) if geslaagd else 0, int(mislukt.group(1)) if mislukt else 0


def schrijf_hoofdstuk(s, cfg, kwaliteit, verkenning, metrics, meta, manifest, belang=None, combinatie=None, tests=(0, 0)):
    doel = meta["doel"]
    test = metrics["test_gekozen_model"]
    base = metrics["test_mediaanbaseline"]
    cv = metrics["cross_validatie"]
    keuze = metrics["gekozen_model"]
    mediaan_train = verkenning["doel_mediaan_kwh"]
    n_test_b, n_test_g = metrics["test_n_buurten"], metrics["test_n_gemeenten"]
    n_train_b, n_train_g = meta["training"]["n_buurten"], meta["training"]["n_gemeenten"]
    sub = metrics["fouten_per_subgroep_test"]

    # Pagina-einde en titel
    s.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    s.kop("Datalab I - Sprint 3 en verder: voorspelmodel elektriciteitslevering", 1)

    # --- Inleiding ---------------------------------------------------
    s.kop("Inleiding en afbakening", 2)
    s.alinea(
        "In sprint 1 richtten wij ons op gasverbruik per postcodegebied. De projectopdracht vraagt echter om "
        "elektriciteit per buurt, en de dataset die wij in sprint 3 via de CBS-API hebben ingeladen (Kerncijfers "
        "wijken en buurten 2023) bevat de elektriciteitslevering per buurt. Daarom maken wij vanaf hier een "
        "duidelijke keuze: het hoofdtraject gaat over elektriciteit per CBS-buurt. Gasverbruik blijft een apart "
        "traject met een eigen doelvariabele; wij gebruiken gasgegevens niet om een elektriciteitsmodel te trainen.")
    s.alinea(
        "Hoe nauwkeurig kunnen we de gemiddelde jaarlijkse elektriciteitslevering per particuliere woning in een "
        "CBS-buurt schatten op basis van beschikbare buurtkenmerken?", vet_begin="Onderzoeksvraag: ")
    s.alinea("Daarbij hebben wij drie doelen:")
    s.opsomming([
        "Elektriciteitslevering vergelijken tussen buurten en gemeenten.",
        "Onderzoeken hoe elektriciteitslevering samenhangt met huishoudensgrootte, woningtype, bouwjaar, "
        "stedelijkheid en het aantal huishoudens.",
        "Een model bouwen en eerlijk testen dat de levering schat op basis van buurtkenmerken, en dit model "
        "beschikbaar maken in een eenvoudige applicatie.",
    ])

    # --- Doeldefinitie -------------------------------------------------
    s.kop("Wat voorspellen we precies?", 2)
    s.alinea(
        "Voordat wij een model bouwden, hebben wij vastgelegd wat de doelvariabele precies betekent. Wij hebben "
        "de officiële CBS-metadata van de tabel gebruikt en niet alleen de kolomnaam " + CBS_DATASET + ".")
    s.tabel("Afspraken over de doelvariabele.", ["Onderdeel", "Afspraak"], [
        ["CBS-kolom", cfg["target_raw_column"]],
        ["Interne naam", cfg["target_internal_name"]],
        ["Betekenis", doel["definitie"]],
        ["Eenheid", doel["eenheid"]],
        ["Waarneming", "Eén CBS-buurt in één jaar"],
        ["Referentiejaar en indeling", f"{doel['referentiejaar']}, wijk- en buurtindeling {doel['referentiejaar']}"],
        ["Sleutel", "buurt_code + target_year"],
        ["Status bij CBS", "Definitief"],
        ["Afronding en geheimhouding", "Afgerond op tientallen kWh; alleen vermeld bij minstens 6 bewoonde woningen"],
        ["Primaire foutmaat", "Ongewogen MAE over buurten"],
    ], breedtes=[4.5, 11.5])
    s.alinea("De schatting is dus:")
    s.opsomming(doel["wat_het_niet_is"])
    s.alinea(
        "Wij voorspellen een gemiddelde per woning, niet het totaal van een buurt. Een totaal krijg je niet door "
        "dit gemiddelde zomaar met het aantal huishoudens te vermenigvuldigen. Woningen, huishoudens, bewoonde "
        "woningen en energieaansluitingen zijn verschillende groepen. Wij doen die omrekening daarom niet.",
        vet_begin="Gemiddelde of totaal? ")

    # --- Bron ----------------------------------------------------------
    b = manifest["bronnen"][0]
    s.kop("Dataset en herkomst", 2)
    s.tabel("Gebruikte bron.", ["Onderdeel", "Waarde"], [
        ["Bron", f"{b['officiele_naam']} (tabel {b['tabelcode']})"],
        ["Periode", b["periode"]],
        ["Geografisch niveau", "Gemeente, wijk en buurt; wij gebruiken alleen buurten"],
        ["Rijen in het ruwe bestand", nl(kwaliteit["rijen_ruw_bestand"])],
        ["Waarvan buurten", nl(kwaliteit["buurten"])],
        ["Laatste wijziging tabel (CBS)", b["bronversie_cbs_gewijzigd"][:10]],
        ["Controle op wijziging", "SHA-256-vingerafdruk van het ongewijzigde bestand (reports/source_manifest.json)"],
    ], breedtes=[5, 11])
    s.alinea(
        "Wij hebben het ruwe bestand ongewijzigd bewaard, samen met de officiële metadata. De precieze ophaaldatum "
        "is niet vastgelegd; de bestandsdatum staat in het manifest. De CBS-energietabellen 86159NED (2024) en "
        "86333NED (2025) hebben wij niet gebruikt. Ze zitten niet in onze projectbestanden en zouden een tweede "
        "jaar en een aparte koppeling vragen. Eén coherent jaar is betrouwbaarder dan jaren en indelingen mengen.")

    # --- Datakwaliteit ---------------------------------------------------
    s.kop("Datavoorbereiding en datakwaliteit", 2)
    s.alinea(
        "Wij filteren eerst op buurten, omdat dit het kleinste en enige consistente niveau is. Als je wijken en "
        "gemeenten laat staan, tel je dezelfde woningen meerdere keren. Daarna hebben wij de volgende controles "
        "uitgevoerd (alle uitkomsten staan in reports/data_quality.json).")
    som = kwaliteit["som_complementaire_percentages"]
    uitval = kwaliteit["uitval_vergelijking"]
    s.tabel("Uitgevoerde controles en uitkomsten.", ["Controle", "Uitkomst"], [
        ["Unieke buurtcodes", f"{nl(kwaliteit['unieke_buurtcodes'])} van {nl(kwaliteit['buurten'])}; 0 dubbel"],
        ["Formaat buurtcode", f"BU + 4 cijfers gemeentecode + 4 tekens. {kwaliteit['buurtcodes_met_letters_in_wijk_of_buurtdeel']} "
                              "codes bevatten letters (bijv. Amsterdam: BU0363AA01), dus codes blijven tekst"],
        ["Gemeentecode uit buurtcode", f"{kwaliteit['gemeenten_in_buurten']} gemeenten; elke code hoort bij precies één gemeentenaam "
                                       "en bestaat ook als gemeenterij in de tabel"],
        ["Percentages tussen 0 en 100", "Geen waarden buiten dit bereik"],
        ["Eengezins + meergezins", f"Telt op tot {nl(som['pct_eengezinswoning + pct_meergezinswoning']['min'])} à "
                                   f"{nl(som['pct_eengezinswoning + pct_meergezinswoning']['max'])}% (afronding)"],
        ["Bouwjaar vóór + vanaf 2000", f"Telt op tot {nl(som['pct_bouwjaar_voor_2000 + pct_bouwjaar_vanaf_2000']['min'])} à "
                                       f"{nl(som['pct_bouwjaar_voor_2000 + pct_bouwjaar_vanaf_2000']['max'])}%"],
        ["Negatieve waarden in gebruikte kolommen", "Geen"],
        ["Buurten met doelwaarde", f"{nl(kwaliteit['buurten_met_doelwaarde'])} (zonder: {nl(kwaliteit['buurten_zonder_doelwaarde'])})"],
        ["Joins", "Niet nodig: doel en kenmerken komen uit dezelfde tabel en indeling"],
    ], breedtes=[5, 11])
    s.alinea(
        f"Buurten zonder gepubliceerde elektriciteitswaarde ({nl(kwaliteit['buurten_zonder_doelwaarde'])}) doen niet mee "
        "aan het trainen. Wij vullen een ontbrekende doelwaarde nooit zelf in. Deze buurten zijn vooral buurten met "
        f"bijna geen woningen: de mediaan is {nl(uitval['mediaan_woningvoorraad_zonder_label'])} woning, en "
        f"{nl(100 * uitval['aandeel_zonder_label_met_0_woningen'])}% heeft helemaal geen woningen (bijvoorbeeld "
        "bedrijventerreinen). Bij buurten mét doelwaarde is de mediaan "
        f"{nl(uitval['mediaan_woningvoorraad_met_label'])} woningen. Het model zegt dus weinig over hele kleine buurten.",
        vet_begin="Uitval. ")
    mis = kwaliteit["missende_kenmerken_binnen_buurten_met_doel"]
    s.alinea(
        f"Bij {nl(mis['pct_meergezinswoning'])} buurten met doelwaarde ontbreken de woningpercentages, omdat het CBS "
        f"die pas vanaf 20 woningen publiceert. De WOZ-waarde ontbreekt bij {nl(mis['woz_waarde_x1000'])} buurten. "
        "Ontbrekende kenmerken vullen wij in met de mediaan van de trainingsdata (zie validatie). Wij behandelen "
        "ontbrekend nooit als nul.", vet_begin="Ontbrekende kenmerken. ")

    # --- Kenmerken -------------------------------------------------------
    s.kop("Keuze van de kenmerken", 2)
    s.alinea(
        "Wij hebben de kenmerken gekozen op basis van onze hypothese uit sprint 1, de 'Relevante kolommen' uit "
        "sprint 3 en de CBS-definities, vóórdat wij de testresultaten kenden. Wij beginnen met een kleine basisset "
        "en vergelijken die met een uitgebreide set.")
    s.tabel("Kenmerken van het model.", ["Interne naam", "CBS-kolom", "Eenheid", "Waarom?", "Kanttekening"], [
        ["pct_meergezinswoning", "PercentageMeergezinswoning_38", "%", "Woningmix: flats/appartementen zijn kleiner",
         "Eengezins is het complement; daarom alleen deze gebruikt"],
        ["pct_bouwjaar_vanaf_2000", "BouwjaarVanaf2000_47", "%", "Nieuwe woningen: andere installaties en isolatie",
         "Geen directe meting van isolatie; complement weggelaten"],
        ["gem_huishoudensgrootte", "GemiddeldeHuishoudensgrootte_33", "personen", "Meer personen, meer apparaten", "Per huishouden, niet per woning"],
        ["stedelijkheid", "MateVanStedelijkheid_125", "code 1-5", "Context: woningtype en -grootte",
         "1 = zeer sterk stedelijk, 5 = niet stedelijk"],
        ["woz_waarde_x1000 (uitgebreid)", "GemiddeldeWOZWaardeVanWoningen_36", "x 1.000 euro",
         "Indirecte aanwijzing voor woninggrootte", "Waardepeildatum 1-1-2022; geen woonoppervlakte"],
        ["pct_koopwoningen (uitgebreid)", "Koopwoningen_41", "%", "Eigendom hangt samen met woningtype en huishouden",
         "Indirect kenmerk, geen oorzaak"],
    ], breedtes=[3.2, 3.8, 1.8, 3.8, 3.8])
    s.alinea(
        "Wij gebruiken nooit als invoer: de doelkolom zelf, elektriciteit per woningtype (Appartement_49 t/m "
        "EigenWoning_55, want dat zijn onderdelen van het doel), gasverbruik, en geografische codes als getal. "
        "Woningvoorraad en aantal huishoudens gebruiken wij alleen in de verkenning, omdat zij de omvang van de buurt "
        "beschrijven en niet het verbruik per woning. Woonoppervlakte zit niet in deze tabel; buurtoppervlakte is "
        "iets anders.", vet_begin="Bewust niet gebruikt. ")

    # --- Validatie ---------------------------------------------------------
    s.kop("Validatieopzet: hoe testen we eerlijk?", 2)
    s.alinea(
        f"Wij hebben de testgroep apart gezet vóórdat wij de data gingen verkennen. Ongeveer 20% van de gemeenten "
        f"is testgemeente ({n_test_g} gemeenten, {nl(n_test_b)} buurten), met GroupShuffleSplit en random_state = 42 "
        f"(scikit-learn developers, 2026a). Alle buurten van een gemeente zitten dus samen in training of samen in "
        f"de test. Buurten in dezelfde gemeente lijken namelijk op elkaar. Als je ze over training en test verdeelt, "
        f"lijkt het model beter dan het is (Roberts et al., 2017). De overige {n_train_g} gemeenten "
        f"({nl(n_train_b)} buurten) zijn trainingsdata.")
    s.alinea(
        "Binnen de trainingsgemeenten vergelijken wij modellen met 5-voudige GroupKFold: steeds wordt een vijfde van "
        "de gemeenten als validatie gebruikt (scikit-learn developers, 2026b). Alle modellen krijgen precies dezelfde "
        "indeling. Invullen van ontbrekende waarden en schalen zitten in een Pipeline. Zo leert het model deze "
        "stappen alleen van de trainingsdata en lekt er geen informatie uit de validatie- of testdata "
        "(scikit-learn developers, 2026c).")
    s.alinea(
        "Wat deze test wél meet: hoe goed het model werkt voor Nederlandse buurten in gemeenten die het niet heeft "
        f"gezien, in {doel['referentiejaar']}. Wat de test níet meet: voorspellingen voor toekomstige jaren, andere "
        "landen, of volledige ruimtelijke onafhankelijkheid (buurgemeenten kunnen nog op elkaar lijken). Daarvoor "
        "zijn meerdere jaren en een andere opzet nodig.")

    # --- Verkenning ----------------------------------------------------------
    s.kop("Verkenning van de trainingsdata", 2)
    s.alinea(
        f"Alle grafieken over buurten gebruiken alleen de {nl(verkenning['n_trainingsbuurten'])} trainingsbuurten. "
        f"De mediaan is {nl(mediaan_train)} kWh per woning per jaar; 80% van de buurten ligt tussen "
        f"{nl(verkenning['doel_p10_p90_kwh'][0])} en {nl(verkenning['doel_p10_p90_kwh'][1])} kWh "
        f"(Figuur {s.volgende_figuur()}).")
    s.figuur("01_verdeling_doel.png", "Verdeling van de gemiddelde elektriciteitslevering per woning (trainingsbuurten, 2023).", 13)
    s.figuur("02_spreiding_kenmerken.png", "Buurtkenmerken tegen elektriciteitslevering per woning (trainingsbuurten).", 16)
    corr = verkenning["correlaties_met_doel"]
    namen = {"pct_meergezinswoning": "Meergezinswoningen (%)", "pct_bouwjaar_vanaf_2000": "Bouwjaar vanaf 2000 (%)",
             "gem_huishoudensgrootte": "Gemiddelde huishoudensgrootte", "woz_waarde_x1000": "WOZ-waarde",
             "pct_koopwoningen": "Koopwoningen (%)", "huishoudens_totaal": "Aantal huishoudens",
             "stedelijkheid": "Stedelijkheid (1 = zeer sterk, 5 = niet)"}
    s.tabel("Rangcorrelatie (Spearman) tussen kenmerk en elektriciteitslevering, trainingsbuurten.",
            ["Kenmerk", "Spearman", "Aantal buurten"],
            [[namen[k], nl(v["spearman"], 2), nl(v["n"])] for k, v in sorted(corr.items(), key=lambda x: -abs(x[1]["spearman"]))],
            breedtes=[7, 3, 3])
    st_med = verkenning["mediaan_doel_per_stedelijkheid"]
    mg_med = verkenning["mediaan_doel_per_meergezinsklasse"]
    s.alinea(
        f"Wat valt op? Buurten met veel koopwoningen, grotere huishoudens en minder stedelijkheid hebben een hogere "
        f"levering per woning. De mediaan loopt op van {nl(st_med['1'])} kWh in zeer sterk stedelijke buurten tot "
        f"{nl(st_med['5'])} kWh in niet-stedelijke buurten. Buurten met 0-10% meergezinswoningen hebben een mediaan "
        f"van {nl(mg_med['0-10'])} kWh, buurten met 90-100% meergezinswoningen {nl(mg_med['90-100'])} kWh. Het aandeel "
        "nieuwbouw hangt maar zwak samen met de levering. Buurten met veel huishoudens hebben vaak een lagere levering "
        "per woning, waarschijnlijk omdat grote buurten vaker stedelijk zijn en meer appartementen hebben.")
    s.alinea(
        "Deze verbanden laten samenhang zien, geen oorzaak. Een hoge WOZ-waarde zorgt niet zelf voor meer "
        "stroomverbruik; het is waarschijnlijk een aanwijzing voor grotere woningen. De kenmerken hangen ook met elkaar "
        "samen: niet-stedelijke buurten hebben vaker eengezinswoningen en koopwoningen.", vet_begin="Let op: ")
    s.figuur("03_stedelijkheid.png", "Elektriciteitslevering per stedelijkheidsklasse (trainingsbuurten).", 12)

    gm = verkenning["gemeenten_cbs"]
    laag = ", ".join(f"{r['gemeente_naam']} ({nl(r['electricity_delivery_kwh_per_dwelling_year'])})" for r in gm["laagste_10"][:4])
    hoog = ", ".join(f"{r['gemeente_naam']} ({nl(r['electricity_delivery_kwh_per_dwelling_year'])})" for r in gm["hoogste_10"][::-1][:4])
    s.kop("Vergelijking tussen gemeenten", 3)
    s.alinea(
        "Voor de gemeentevergelijking gebruiken wij de cijfers die het CBS zelf per gemeente publiceert. Een gewoon "
        "gemiddelde van buurtgemiddelden zou de gemiddelde buurt beschrijven, niet de gemiddelde woning. Van de "
        f"{gm['aantal_met_cijfer']} gemeenten is de mediaan {nl(gm['mediaan_kwh'])} kWh per woning. Het laagst zijn "
        f"grote steden: {laag} kWh. Het hoogst: {hoog} kWh (Figuur {s.volgende_figuur()}). Dit past bij het patroon "
        "uit sprint 1: gemeenten met veel grote, vrijstaande woningen verbruiken meer. Deze gemeentecijfers zijn "
        "beschrijvend en zijn niet gebruikt om het model te kiezen.")
    s.figuur("06_gemeenten_laag_hoog.png", "De 10 gemeenten met de laagste en de hoogste gemiddelde elektriciteitslevering per woning (CBS-gemeentecijfers, 2023).", 14)

    # --- Modelvergelijking ------------------------------------------------------
    s.kop("Modelvergelijking", 2)
    s.alinea(
        "Wij vergelijken drie soorten modellen, elk met de basis- en de uitgebreide kenmerkenset (Pedregosa et al., 2011):")
    s.opsomming([
        "Mediaanbaseline: voorspelt altijd de mediaan van de training. Een model moet hier duidelijk beter zijn.",
        "Ridge-regressie: een lineair model dat goed uit te leggen is.",
        "Beperkt Random Forest: veel beslisbomen samen (200 bomen, maximaal 8 niveaus, minstens 20 buurten per blad). "
        "Dit model kan niet-lineaire verbanden vinden.",
    ])
    volgorde = ["mediaanbaseline", "ridge_basis", "ridge_uitgebreid", "random_forest_basis", "random_forest_uitgebreid"]
    s.tabel("Resultaten cross-validatie (5 folds, trainingsgemeenten). Gemiddelde ± standaardafwijking over de folds.",
            ["Model", "MAE (kWh)", "RMSE (kWh)", "R²"],
            [[n.replace("_", " "), f"{nl(cv[n]['MAE_gemiddeld'])} ± {nl(cv[n]['MAE_sd'])}",
              f"{nl(cv[n]['RMSE_gemiddeld'])} ± {nl(cv[n]['RMSE_sd'])}",
              f"{nl(cv[n]['R2_gemiddeld'], 2)} ± {nl(cv[n]['R2_sd'], 2)}"] for n in volgorde],
            breedtes=[5, 3.5, 3.5, 3])
    s.alinea(
        f"Volgens onze vooraf gekozen regel (laagste gemiddelde MAE in de cross-validatie) kiezen wij "
        f"{keuze.replace('_', ' ')}. Twee dingen helpen: de uitgebreide kenmerken (WOZ-waarde en koopwoningen) en "
        f"het Random Forest, dat ook gebogen verbanden kan volgen. Ridge met de uitgebreide set is wel goed uit te "
        f"leggen en we gebruiken het hieronder om de richting van de verbanden te laten zien.")
    s.figuur("09_modelvergelijking_cv.png", "Gemiddelde MAE per model in de cross-validatie (lager is beter).", 14)

    # --- Testresultaten ------------------------------------------------------
    s.kop("Resultaten op de testgemeenten", 2)
    verbetering = metrics["test_verbetering_mae_tov_baseline_pct"]
    s.alinea(
        f"Pas na de keuze hebben wij het model één keer getest op de {n_test_g} testgemeenten.")
    s.tabel("Testresultaten (één keer berekend, na de modelkeuze).", ["Model", "MAE (kWh)", "RMSE (kWh)", "R²"], [
        [keuze.replace("_", " "), nl(test["MAE"]), nl(test["RMSE"]), nl(test["R2"], 2)],
        ["mediaanbaseline", nl(base["MAE"]), nl(base["RMSE"]), nl(base["R2"], 2)],
    ], breedtes=[5, 3.5, 3.5, 3])
    s.alinea("Wat betekenen deze maten?")
    s.opsomming([
        f"MAE {nl(test['MAE'])} kWh: gemiddeld zit de schatting {nl(test['MAE'])} kWh per woning per jaar naast de "
        f"CBS-waarde. Dat is ongeveer {nl(100 * test['MAE'] / mediaan_train)}% van de mediaan ({nl(mediaan_train)} kWh). "
        f"De baseline zit gemiddeld {nl(base['MAE'])} kWh ernaast; het model is {nl(verbetering)}% beter.",
        f"RMSE {nl(test['RMSE'])} kWh: hoger dan de MAE, omdat grote fouten zwaarder tellen. Er zijn dus buurten "
        "met een flink grotere fout.",
        f"R² {nl(test['R2'], 2)}: het model verklaart een groot deel van de verschillen tussen buurten, vergeleken met "
        "steeds hetzelfde getal voorspellen. R² is geen percentage nauwkeurigheid (scikit-learn developers, 2026d).",
    ])
    s.figuur("07_test_werkelijk_vs_voorspeld.png", "Werkelijke en voorspelde levering in de testgemeenten. Punten op de stippellijn zijn perfect voorspeld.", 11)
    s.alinea(
        "In Figuur " + str(s.figuur_start + s.figuur_nr) + " is te zien dat het model hoge waarden vaak iets te laag "
        "schat en lage waarden iets te hoog. Het model trekt extreme buurten dus naar het midden.")

    rijen = []
    for k, v in sub["stedelijkheid"].items():
        rijen.append([f"Stedelijkheid {int(float(k))}", nl(v["n"]), nl(v.get("MAE", float("nan"))), nl(v.get("R2", float("nan")), 2)])
    rijen.append(["Alle kenmerken bekend", nl(sub["kenmerk_ontbreekt"]["False"]["n"]),
                  nl(sub["kenmerk_ontbreekt"]["False"]["MAE"]), nl(sub["kenmerk_ontbreekt"]["False"]["R2"], 2)])
    rijen.append(["Minstens één kenmerk ontbreekt", nl(sub["kenmerk_ontbreekt"]["True"]["n"]),
                  nl(sub["kenmerk_ontbreekt"]["True"]["MAE"]), nl(sub["kenmerk_ontbreekt"]["True"]["R2"], 2)])
    s.tabel("Testfout per subgroep.", ["Groep", "Buurten", "MAE (kWh)", "R²"], rijen, breedtes=[6, 3, 3, 3])
    s.alinea(
        f"De fout is het kleinst in zeer sterk stedelijke buurten en het grootst in niet-stedelijke buurten, waar "
        f"de levering ook meer varieert. Buurten waarbij een kenmerk ontbreekt hebben een veel grotere fout "
        f"(MAE {nl(sub['kenmerk_ontbreekt']['True']['MAE'])} kWh). Voor die buurten is de schatting weinig betrouwbaar; "
        "de app waarschuwt hiervoor.")

    # --- Interpretatie Ridge ---------------------------------------------------
    s.kop("Uitleg van de verbanden (Ridge-model)", 3)
    coef = {c["kenmerk"]: c for c in meta["ridge_uitleg"]["coefficienten"]}
    uitleg = [
        ("gem_huishoudensgrootte", "+1 persoon per huishouden"),
        ("stedelijkheid", "1 klasse minder stedelijk"),
        ("woz_waarde_x1000", "+100.000 euro WOZ-waarde", 100),
        ("pct_koopwoningen", "+10 procentpunt koopwoningen", 10),
        ("pct_meergezinswoning", "+10 procentpunt meergezinswoningen", 10),
        ("pct_bouwjaar_vanaf_2000", "+10 procentpunt nieuwbouw", 10),
    ]
    rijen = []
    for item in uitleg:
        kenmerk, omschrijving = item[0], item[1]
        factor = item[2] if len(item) > 2 else 1
        if kenmerk in coef:
            rijen.append([omschrijving, f"{'+' if coef[kenmerk]['effect_per_eenheid_kwh'] * factor >= 0 else ''}"
                                        f"{nl(coef[kenmerk]['effect_per_eenheid_kwh'] * factor)} kWh"])
    s.tabel("Geschat verschil in levering per woning volgens het Ridge-model met de uitgebreide set, als de andere kenmerken gelijk blijven.",
            ["Verandering", "Verschil in voorspelling"], rijen, breedtes=[8, 5])
    s.alinea(
        "Huishoudensgrootte, stedelijkheid en WOZ-waarde hebben het grootste verband. Meergezinswoningen en nieuwbouw "
        "hebben in dit model maar een klein extra verband, omdat hun informatie al grotendeels in de andere kenmerken "
        "zit. Dit zijn verbanden binnen het model, geen oorzaken.")

    # --- App -------------------------------------------------------------------
    s.kop("De applicatie", 2)
    s.alinea(
        "Wij hebben een Nederlandstalige Streamlit-app gebouwd (Streamlit, z.d.). De app gebruikt exact hetzelfde "
        "opgeslagen model (artifacts/pipeline.joblib) als de batchvoorspelling. Bij het ontwerp hebben wij gekeken "
        "naar vergelijkbare openbare hulpmiddelen, zoals de Klimaatmonitor en CBS-kaarten (zoeken op een plek, "
        "vergelijken met gemeente en Nederland, downloaden), en naar aanbevelingen voor het uitleggen van "
        "voorspelmodellen. De app heeft vijf tabbladen:")
    s.opsomming([
        "Schatting: een formulier met zes buurtkenmerken en een verzendknop, met voorbeelden van typische buurten "
        "(mediaanwaarden van trainingsbuurten per stedelijkheidsklasse). De uitkomst staat met eenheid, jaar en "
        "modelversie, op een 'leveringsmeter' die laat zien waar de schatting valt tussen alle trainingsbuurten, "
        "plus de testfout met uitleg. Een 'wat als'-grafiek laat zien hoe de schatting van het model verandert als "
        "één kenmerk anders is. 'Waarom deze schatting?' toont per kenmerk hoeveel het de schatting omhoog of omlaag "
        "duwt (SHAP-waarden; Lundberg & Lee, 2017; Lundberg et al., 2020), als grafiek én als tabel. Gebruikers "
        "kunnen hun eigen jaarlijkse levering ernaast zetten; de app zegt er dan bij dat één huishouden iets anders "
        "is dan een buurtgemiddelde. De schatting is te downloaden, en een CSV met veel buurten tegelijk kan worden "
        "geüpload (zelfde controles als de batchvoorspelling).",
        "Buurt: zoeken op postcode (via de PDOK Locatieserver; PDOK, z.d.) of zelf een buurt kiezen. CBS-waarde "
        "naast de schatting, met de melding of de buurt in de training of de test zat. Een goede schatting voor een "
        "trainingsbuurt bewijst niets over nieuwe buurten. Voor buurten zonder woningen maakt de app geen schatting. "
        "Een tabel zet de buurt naast haar gemeente, heel Nederland en eventueel een tweede buurt.",
        "Gemeenten: het CBS-gemeentecijfer, de plek tussen alle gemeenten (gelijke cijfers delen een plek) en de "
        "spreiding van de buurten binnen de gemeente. Dit beantwoordt doel 1 (vergelijken).",
        "Model: testfout, vergelijking van de modellen, fout per groep, kenmerkbelang, de controle op ongewone "
        "combinaties en een download met schattingen voor alle buurten.",
        "Uitleg: een modelkaart volgens Mitchell et al. (2019): wat het model meet, waarvoor het wel en niet bedoeld "
        "is, gegevens, evaluatie en beperkingen. De kaart wordt uit de modelbestanden gemaakt (MODEL_CARD.md).",
    ])
    s.alinea(
        "De invoer wordt gecontroleerd: het moeten eindige getallen zijn, percentages liggen tussen 0 en 100 en "
        "stedelijkheid is een klasse van 1 tot 5. Een leeg veld mag: het model behandelt het als onbekend en de "
        "app waarschuwt dat de schatting dan minder betrouwbaar is. Helemaal niets invullen geeft een foutmelding. "
        "Waarden buiten het bereik van de trainingsdata geven een waarschuwing, en ook een combinatie van waarden "
        "die op geen enkele trainingsbuurt lijkt (zie hieronder). De app toont geen "
        "betrouwbaarheidsinterval, omdat wij dat niet apart hebben onderzocht; de test-MAE is geen garantie voor één "
        "buurt. Bij het ontwerp hebben wij gelet op leesbaarheid: alle tekst haalt een contrast van minstens 4,5:1, "
        "getallen staan in Nederlandse notatie en de app werkt ook op een telefoonscherm.")
    s.figuur("11_app_schatting.jpg", "Schermafbeelding van de app na een schatting (typische weinig stedelijke buurt).", 15)
    if os.path.exists(os.path.join(PROJECT_DIR, "reports", "figures", "12_app_uitleg.jpg")):
        s.figuur("12_app_uitleg.jpg", "Tabblad Buurt na zoeken op postcode, met de uitleg per kenmerk (SHAP).", 15)

    # --- Uitleg en controles ---------------------------------------------------------
    if belang and combinatie:
        s.kop("Uitleg van het Random Forest en controle op ongewone invoer", 3)
        s.alinea(
            "Om te zien op welke kenmerken het gekozen model leunt, hebben wij op de testgemeenten steeds één kenmerk "
            "door elkaar gehusseld en gemeten hoeveel de MAE dan stijgt (permutation importance, 10 herhalingen; "
            "scikit-learn developers, 2026e). Kenmerken die samenhangen, zoals WOZ-waarde, koopwoningen en "
            "meergezinswoningen, kunnen elkaar deels vervangen; hun belang wordt dan te laag ingeschat.")
        volgorde = sorted(belang["kenmerken"].items(), key=lambda kv: -kv[1]["toename_mae_gemiddeld"])
        s.tabel("Toename van de test-MAE als één kenmerk wordt gehusseld (gemiddelde en standaardafwijking over 10 keer).",
                ["Kenmerk", "Toename MAE (kWh)", "Spreiding (kWh)"],
                [[namen.get(k, k), nl(v["toename_mae_gemiddeld"]), nl(v["toename_mae_sd"], 1)] for k, v in volgorde], breedtes=[7, 4, 4])
        c = combinatie
        s.alinea(
            "Elk kenmerk apart kan normaal zijn terwijl de combinatie ongewoon is. Daarom leert de app met Local "
            f"Outlier Factor (Breunig et al., 2000) hoe de {nl(c['train_n_compleet'])} trainingsbuurten met alle "
            f"kenmerken bekend eruitzien. De grens ligt bij de {nl(c['grens_pct'])}% meest afwijkende combinaties; "
            "de controle werkt alleen als alle zes kenmerken bekend zijn. In de testgemeenten ging de waarschuwing af "
            f"bij {c['test_n_gemarkeerd']} van {nl(c['test_n_compleet'])} buurten. "
            + (f"Hun gemiddelde fout was {nl(c['test_mae_gemarkeerd'])} kWh, tegen {nl(c['test_mae_overig'])} kWh "
               "bij de andere buurten." if c["test_n_gemarkeerd"] >= 30 else
               "Dat zijn er minder dan 30; volgens onze eigen regel is dat te weinig om hun fout apart te rapporteren. "
               + ("In deze kleine groep was de fout wel groter dan bij de andere buurten. "
                  if (c["test_mae_gemarkeerd"] or 0) > c["test_mae_overig"] else "")
               + "De waarschuwing zegt daarom alleen dat de invoer ongewoon is, niet hoe groot de fout dan is."))

    # --- Controle ----------------------------------------------------------------
    s.kop("Controle en reproduceerbaarheid", 2)
    s.alinea(
        "Wij hebben de hele analyse opnieuw uitgevoerd vanaf het ruwe bestand. De indeling en alle resultaten waren "
        f"precies hetzelfde. Het opgeslagen model geeft na opnieuw laden dezelfde voorspellingen (grootste verschil "
        f"{meta['herlaadcontrole']['max_verschil_kwh']:.0e} kWh). De acceptatiecriteria zijn vastgelegd als "
        "automatische tests (tests/test_acceptatie.py, tests/test_app.py en tests/test_uitbreidingen.py); "
        + (f"alle {tests[0]} tests slagen." if tests[1] == 0 else f"{tests[0]} tests slagen en {tests[1]} niet."))
    s.tabel("Acceptatiecriteria en bewijs.", ["Criterium", "Bewijs"], [
        ["Doeldefinitie volgt CBS-metadata", "Test op metadata; tabel 'Afspraken over de doelvariabele'"],
        ["Unieke buurt-jaar-sleutels", "Test + data_quality.json"],
        ["Geen rijvermenigvuldiging door joins", "Geen joins; aantal rijen = aantal buurten (test)"],
        ["Doel niet geïmputeerd", "Test: buurten zonder CBS-waarde zitten niet in train/test"],
        ["Gemeenten gescheiden", "Test: geen gemeente in train én test; één fold per gemeente"],
        ["Voorbereiding alleen op trainingsdata", "Test: invulwaarden = mediaan van trainingsdata"],
        ["Zelfde folds voor alle modellen", "Test op metrics.json"],
        ["Maten reproduceerbaar", "Test herberekent MAE, RMSE en R² uit test_predictions.csv"],
        ["Herladen model gelijk", "Test + herlaadcontrole in train.py"],
        ["App behandelt geldige, lege, ongeldige en afwijkende invoer", "Tests op de voorspelroute en op de app zelf (Streamlit AppTest) + controle in de browser op desktop en mobiel"],
        ["README maakt uitvoering reproduceerbaar", "README.md met stappen; volledige herhaling uitgevoerd"],
        ["Rapport en app noemen beperkingen en periode", "Dit hoofdstuk, tabblad 'Uitleg' en MODEL_CARD.md"],
    ], breedtes=[7, 9])

    # --- Beperkingen ---------------------------------------------------------------
    s.kop("Beperkingen", 2)
    s.opsomming([
        f"Eén jaar ({doel['referentiejaar']}): het model is geen voorspelling voor toekomstige jaren.",
        "Levering uit het net is niet het volledige verbruik: woningen met zonnepanelen gebruiken een deel van hun "
        "eigen stroom direct. Buurten met veel zonnepanelen kunnen daardoor lager uitkomen.",
        "Kleine buurten ontbreken vaker of hebben minder kenmerken. Juist daar is het model het minst betrouwbaar.",
        "CBS rondt de energiecijfers af op tientallen kWh en aantallen op buurtniveau op vijftallen.",
        "WOZ-waarde en koopwoningen zijn indirecte kenmerken. Echte woonoppervlakte en isolatie zaten niet in de data.",
        "De test meet overdraagbaarheid naar ongeziene gemeenten, maar buurgemeenten kunnen nog steeds op elkaar lijken.",
        "Jaarlijkse kWh zeggen niets over piekbelasting of netcongestie op een bepaald moment.",
    ])

    # --- Conclusie -------------------------------------------------------------------
    s.kop("Conclusie", 2)
    s.alinea(
        f"Met zes buurtkenmerken uit de CBS-kerncijfers kunnen wij de gemiddelde jaarlijkse elektriciteitslevering "
        f"per woning in buurten van ongeziene gemeenten schatten met een gemiddelde fout van ongeveer "
        f"{nl(round(test['MAE'], -1))} kWh per woning per jaar (R² {nl(test['R2'], 2)}). Dat is "
        f"{nl(verbetering)}% beter dan steeds de mediaan voorspellen. Huishoudensgrootte, stedelijkheid, WOZ-waarde "
        f"en het aandeel koopwoningen hangen het sterkst samen met de levering. Voor kleine buurten en buurten met "
        f"ontbrekende gegevens is de schatting veel minder nauwkeurig. Het model geldt voor {doel['referentiejaar']} "
        "en beschrijft samenhang, geen oorzaken.")

    s.kop("Vervolgstappen (plan, nog niet uitgevoerd)", 2)
    s.opsomming([
        "Meer jaren toevoegen (bijvoorbeeld CBS 86159NED en 86333NED) met dezelfde definitie, en dan testen op een later jaar.",
        "Het gastraject als apart model uitwerken, met een eigen datacontract.",
        "Een scenarioanalyse voor warmtepompen opzetten, met duidelijk vastgelegde aannames. Voor netcongestie zijn "
        "tijdprofielen en netgegevens nodig, niet alleen jaarlijkse kWh.",
    ])

    # --- Rubric ---------------------------------------------------------------------
    s.kop("Koppeling met het beoordelingsformulier sprint 3", 2)
    s.alinea(
        "Het beoordelingsformulier van sprint 3 gaat over het notebook. Hieronder staat waar elk onderdeel wordt "
        "behandeld. Wij geven onszelf geen score; dat doet de docent.")
    s.tabel("Rubriconderdelen en waar ze aan bod komen.", ["Rubriconderdeel", "Waar?"], [
        ["Dataset importeren en filteren", "Notebook (CBS-API, filter op buurten); dit hoofdstuk: 'Datavoorbereiding' (granulariteit)"],
        ["Beschrijven van de dataset", "Notebook (kolomtabel en metrics); reports/source_manifest.json met CBS-definities en eenheden"],
        ["Kolommen verwijderen", "Notebook (40%-threshold met bron); dit hoofdstuk: 'Keuze van de kenmerken' en 'Bewust niet gebruikt'"],
        ["Basisstatistieken", "Notebook (totalen, percentages, vergelijking CBS); dit hoofdstuk: gemeentevergelijking"],
        ["Vormeisen van het notebook", "Notebook; code in src/ heeft kopjes en commentaar"],
        ["Samenwerken, projectleiderschap, planning", "Reflectie en GenAI-evaluatie hieronder (door de groep aan te vullen)"],
    ], breedtes=[5, 11])

    # --- GenAI en reflectie ------------------------------------------------------------
    s.kop("GenAI-impactevaluatie (dit onderdeel)", 2)
    s.alinea(
        "Voor dit onderdeel hebben wij Claude (Anthropic) gebruikt als hulp bij het schrijven van de Python-code, "
        "het opzetten van de controles en tests, en een eerste versie van deze tekst. Alle getallen in dit hoofdstuk "
        "komen rechtstreeks uit de bestanden die de code maakt; ze zijn niet door de AI bedacht.")
    s.aanvullen("wat jullie zelf hebben gecontroleerd (bijv. code doorgelopen, app getest, CBS-definities nagelezen), "
                "wat jullie hebben aangepast en welke aandachtspunten jullie meenemen naar de volgende sprint.")
    s.kop("Reflectie", 2)
    s.aanvullen("taakverdeling, samenwerking, een praktijkvoorbeeld per onderdeel en een verbetervoorstel (zie rubric 'Projectleiderschap').")


# ----------------------------------------------------------------------
# Bronnen
# ----------------------------------------------------------------------
NIEUWE_BRONNEN = [
    f"{CBS}. (2026e). Kerncijfers wijken en buurten 2023 [Dataset, tabel 85618NED, gewijzigd 31 maart 2026]. StatLine. "
    "https://opendata.cbs.nl/ODataApi/OData/85618NED",
    "Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying density-based local outliers. "
    "In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data (pp. 93–104). "
    "https://doi.org/10.1145/342009.335388",
    "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., "
    "& Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. "
    "Nature Machine Intelligence, 2(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in "
    "Neural Information Processing Systems 30 (pp. 4765–4774).",
    "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & "
    "Gebru, T. (2019). Model cards for model reporting. In Proceedings of the Conference on Fairness, "
    "Accountability, and Transparency (pp. 220–229). https://doi.org/10.1145/3287560.3287596",
    "PDOK. (z.d.). Locatieserver (versie 3.1). Geraadpleegd op 10 november 2026, van "
    "https://api.pdok.nl/bzk/locatieserver/search/v3_1/ui/",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., "
    "Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. "
    "(2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., "
    "Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. "
    "(2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. "
    "Ecography, 40(8), 913–929. https://doi.org/10.1111/ecog.02881",
    "scikit-learn developers. (2026a). GroupShuffleSplit (versie 1.9). Geraadpleegd op 10 november 2026, van "
    "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html",
    "scikit-learn developers. (2026b). GroupKFold (versie 1.9). Geraadpleegd op 10 november 2026, van "
    "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html",
    "scikit-learn developers. (2026c). Common pitfalls and recommended practices (versie 1.9). Geraadpleegd op "
    "10 november 2026, van https://scikit-learn.org/stable/common_pitfalls.html",
    "scikit-learn developers. (2026d). Metrics and scoring: Regression metrics (versie 1.9). Geraadpleegd op "
    "10 november 2026, van https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics",
    "scikit-learn developers. (2026e). Permutation feature importance (versie 1.9). Geraadpleegd op 10 november "
    "2026, van https://scikit-learn.org/stable/modules/permutation_importance.html",
    "Streamlit. (z.d.). st.form. Geraadpleegd op 10 november 2026, van "
    "https://docs.streamlit.io/develop/api-reference/execution-flow/st.form",
]


def voeg_bronnen_toe(doc, bronnen_kop):
    """Zet nieuwe bronnen alfabetisch tussen de bestaande, met dezelfde opmaak."""
    def bestaande_bronnen():
        """Alle niet-lege alinea's ná de kop Bronnenlijst."""
        resultaat, na_kop = [], False
        for p in doc.paragraphs:
            if p._p is bronnen_kop._p:
                na_kop = True
            elif na_kop and p.text.strip():
                resultaat.append(p)
        return resultaat

    bestaande = bestaande_bronnen()
    if not bestaande:
        raise SystemExit("Geen bestaande bronnen gevonden.")
    sjabloon = bestaande[0]
    for tekst in NIEUWE_BRONNEN:
        # Kopie van een bestaande bron (zelfde opmaak), zonder de tekst
        nieuw = copy.deepcopy(sjabloon._p)
        for kind in list(nieuw):
            if kind.tag.endswith("}r") or kind.tag.endswith("}hyperlink"):
                nieuw.remove(kind)
        volgende = next((p for p in bestaande if p.text.strip().casefold() > tekst.casefold()), None)
        if volgende is not None:
            volgende._p.addprevious(nieuw)
        else:
            bestaande[-1]._p.addnext(nieuw)
        Paragraph(nieuw, sjabloon._parent).add_run(tekst)
        bestaande = bestaande_bronnen()


def main():
    cfg = lees("config.json")
    kwaliteit = lees("reports/data_quality.json")
    verkenning = lees("reports/exploration_summary.json")
    metrics = lees("reports/metrics.json")
    meta = lees("artifacts/model_metadata.json")
    manifest = lees("reports/source_manifest.json")
    belang = lees("reports/feature_importance.json")
    combinatie = lees("reports/combinatiecheck.json")
    tests = testuitslag()
    print(f"Tests: {tests[0]} geslaagd, {tests[1]} mislukt")

    doc = Document(BRON)
    bronnen_kop = next(p for p in doc.paragraphs if p.text.strip() == "Bronnenlijst" and p.style.name.startswith("Heading"))

    # Nieuwe inhoud eerst aan het eind toevoegen, daarna vóór de Bronnenlijst verplaatsen.
    body = doc.element.body
    voor_lijst = list(body.iterchildren())  # lijst bewaren, zodat id()'s uniek blijven
    voor = set(id(e) for e in voor_lijst)
    schrijf_hoofdstuk(Schrijver(doc), cfg, kwaliteit, verkenning, metrics, meta, manifest, belang, combinatie, tests)
    nieuw = [e for e in body.iterchildren() if id(e) not in voor and not e.tag.endswith("}sectPr")]
    for e in nieuw:
        bronnen_kop._p.addprevious(e)

    voeg_bronnen_toe(doc, bronnen_kop)
    doc.save(DOEL)
    print(f"Geschreven: {os.path.relpath(DOEL, PROJECT_DIR)} ({len(nieuw)} nieuwe blokken, {len(NIEUWE_BRONNEN)} bronnen)")


if __name__ == "__main__":
    main()
