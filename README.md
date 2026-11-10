# DataLab: elektriciteitslevering per woning in Nederlandse buurten

Groep 4, DataLab I.

## Onderzoeksvraag

> Hoe nauwkeurig kunnen we de gemiddelde jaarlijkse elektriciteitslevering per particuliere woning in een CBS-buurt schatten op basis van beschikbare buurtkenmerken?

## Wat voorspellen we precies?

| Onderdeel | Afspraak |
|---|---|
| Doel | `electricity_delivery_kwh_per_dwelling_year` (CBS-kolom `GemiddeldeElektriciteitsleveringTotaal_48`) |
| Betekenis | Gemiddelde jaarlijkse elektriciteitslevering **uit het openbare net** op individuele aansluitingen van particuliere woningen, alle woningtypen samen |
| Eenheid | kWh per woning per jaar |
| Waarneming | Eén CBS-buurt in 2023 (wijk- en buurtindeling 2023) |
| Sleutel | `buurt_code` + `target_year` |
| Bron | CBS 85618NED, *Kerncijfers wijken en buurten 2023* (status: definitief) |
| Primaire foutmaat | Ongewogen MAE over buurten |

Het model voorspelt **niet**: het volledige verbruik achter de meter (direct gebruikte eigen opwek telt niet volledig mee), het totale energiegebruik van een buurt, het verbruik van één huishouden of piekvermogen/netcongestie. Collectieve levering (lift, galerijverlichting) zit er niet in. CBS rondt af op tientallen kWh en publiceert het cijfer alleen bij minstens 6 bewoonde woningen.

## Projectstructuur

```text
README.md, requirements.txt, config.json
app.py                         Streamlit-app
data/raw/                      ongewijzigd CBS-bestand (85618NED)
data/metadata/                 officiële CBS-metadata (DataProperties, TableInfos)
data/processed/model_table.csv één rij per buurt (interne kolomnamen)
data/processed/gemeente_table.csv  door CBS gepubliceerde gemeentecijfers
data/processed/regio_kenmerken.csv kenmerken van gemeenten en Nederland (alleen om te vergelijken)
src/prepare_data.py            stap 1: controleren en modeltabel maken
src/split_data.py              stap 2: testgemeenten en folds vastleggen
src/explore.py                 stap 3: verkenning (alleen trainingsdata)
src/train.py                   stap 4: modellen vergelijken, testen, opslaan
src/predict.py                 stap 5: voorspellen (app en batch), combinatiecontrole, uitleg per schatting (SHAP)
src/explain.py                 stap 6: kenmerkbelang, combinatiecontrole leren en testen, modelkaart
src/model_card.py              tekst van de modelkaart (gebruikt door app en explain.py)
src/postcode.py                postcode -> buurt via de PDOK Locatieserver
src/update_report.py           stap 7: nieuw hoofdstuk in het Word-rapport (getallen uit reports/)
MODEL_CARD.md                  modelkaart (Mitchell et al., 2019), gegenereerd door explain.py
report/                        Word-rapporten (alleen lokaal, staan niet op GitHub)
tests/test_acceptatie.py       bewijs voor de acceptatiecriteria
tests/test_app.py              de app zelf (Streamlit AppTest)
tests/test_uitbreidingen.py    uitleg, combinatiecontrole, modelkaart, postcode, CSV
artifacts/pipeline.joblib      opgeslagen voorbereiding + model
artifacts/model_metadata.json  definitie, kenmerken, bereik, testfout, versies
artifacts/combinatiecheck.joblib  detector voor ongewone combinaties van kenmerken
reports/source_manifest.json   herkomst, hash en definities van de bron
reports/data_quality.json      rijaantallen, ontbrekende waarden, uitval
reports/split_membership.csv   per buurt: train/test en fold
reports/metrics.json           cross-validatie en testresultaten
reports/test_predictions.csv   werkelijke en voorspelde testwaarden
reports/feature_importance.json  kenmerkbelang op de testgemeenten
reports/combinatiecheck.json   hoe vaak de combinatiecontrole afgaat, en de fout daarbij
reports/exploration_summary.json  getallen bij de verkenning
reports/figures/               grafieken
```

## Installeren (eenmalig)

Nodig: Python 3.12 of nieuwer (getest met 3.14).

```bash
cd DataLab_elektriciteitsmodel
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows: gebruik `.venv\Scripts\pip` en `.venv\Scripts\python` in plaats van `.venv/bin/...`.

## Alles opnieuw uitvoeren

```bash
.venv/bin/python src/prepare_data.py
.venv/bin/python src/split_data.py
.venv/bin/python src/explore.py
.venv/bin/python src/train.py
.venv/bin/python src/explain.py
.venv/bin/python -m pytest -v
.venv/bin/python src/update_report.py      # Word-rapport bijwerken -> report/Data Lab-5.docx
```

Met dezelfde ruwe data, `config.json` en softwareversies krijg je dezelfde indeling en dezelfde resultaten (`random_state = 42`).

## De app starten

```bash
.venv/bin/streamlit run app.py --server.address localhost
```

Open daarna <http://localhost:8501>. Tabbladen:

1. **Schatting**: vul buurtkenmerken in of kies een typische buurt. Lege velden mogen (het model behandelt ze als onbekend), maar geven een waarschuwing. Ongewone combinaties van kenmerken geven ook een waarschuwing. Daarna: je eigen verbruik ernaast zetten, "Waarom deze schatting?" (bijdrage per kenmerk, SHAP) en de schatting downloaden. Onderaan: een CSV met veel buurten tegelijk uploaden en het resultaat downloaden.
2. **Buurt**: zoek op postcode of kies zelf een buurt. CBS-waarde naast de schatting, of de buurt in training of test zat, je eigen verbruik, de uitleg per kenmerk en een vergelijking met de gemeente, Nederland en een tweede buurt.
3. **Gemeenten**: CBS-gemeentecijfer, plek tussen alle gemeenten, spreiding van de buurten en de ranglijst als CSV.
4. **Model**: testfout, modelvergelijking, fout per groep, kenmerkbelang, de combinatiecontrole en een CSV met schattingen voor alle buurten.
5. **Uitleg**: de modelkaart (wat het model wel en niet is, gegevens, evaluatie, beperkingen), ook te downloaden.

De postcodezoeker heeft internet nodig (PDOK Locatieserver); de postcode wordt alleen daarheen gestuurd en niet bewaard. Zonder internet kies je de buurt zelf. Downloads gebruiken een puntkomma en een decimale komma, zodat Nederlandse Excel ze goed opent.

Het thema staat in `.streamlit/config.toml`; ontwerpkeuzes, contrastwaarden en schermafbeeldingen staan in `reports/design/` (controle: `.venv/bin/python reports/design/check_evidence.py reports/design/evidence.json`).

Bekend: Streamlit 1.58 kan in de terminal meldingen `Exception in ASGI application` tonen bij het serveren van eigen bestanden. Dit komt uit Streamlit zelf; de app werkt gewoon.

## Batchvoorspelling

```bash
.venv/bin/python src/predict.py --input mijn_buurten.csv --output voorspellingen.csv
```

Het invoerbestand heeft kolommen `pct_meergezinswoning`, `pct_bouwjaar_vanaf_2000`, `gem_huishoudensgrootte`, `stedelijkheid` (1 t/m 5), `woz_waarde_x1000` en `pct_koopwoningen`. De batch gebruikt dezelfde invoercontroles als de app.

## Werkwijze in het kort

1. Alleen buurten (geen wijken/gemeenten), zodat er geen dubbeltellingen zijn.
2. Buurten zonder gepubliceerde doelwaarde (968) vallen af; de doelwaarde wordt **nooit** ingevuld.
3. ~20% van de **gemeenten** (69) is vóór de verkenning apart gezet als testgroep (`GroupShuffleSplit`). Binnen de 273 trainingsgemeenten: 5 folds (`GroupKFold`).
4. Invullen van ontbrekende kenmerken en schalen gebeurt in een `Pipeline`, dus alleen op trainingsdata geleerd.
5. Vergeleken: mediaanbaseline, Ridge en beperkt Random Forest, elk met een basis- en uitgebreide kenmerkenset.
6. Gekozen op laagste gemiddelde CV-MAE, daarna één keer getest.

Resultaten staan in `reports/metrics.json` en worden in het Word-rapport besproken.
