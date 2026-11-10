# Modelkaart: elektriciteitslevering per woning (CBS-buurt, 2023)

## Modeldetails

- **Versie:** neighbourhood_electricity_delivery-20261110 (getraind op 10 november 2026)
- **Type:** Random Forest-regressie (200 bomen, maximale diepte 8, minimale bladgrootte 20), gekozen uit 4 kandidaten (plus een mediaanbaseline als referentie) op de laagste MAE in 5-voudige cross-validatie.
- **Voorbewerking:** een ontbrekend kenmerk krijgt de mediaan van de training. Voor kenmerken die in de training soms ontbraken komt er ook een 0/1-kolom 'ontbrak' bij (niet voor stedelijkheid, dat in de training nooit ontbrak). Daarna standaardiseren. Alles zit in één opgeslagen scikit-learn-pipeline.
- **Software:** Python 3.14.6, scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6. Een pipeline-bestand werkt alleen betrouwbaar met dezelfde scikit-learn-versie.
- **Gemaakt door:** DataLab I, groep 4 (studentenproject).
- **Licentie:** de CBS-gegevens vallen onder CC BY 4.0 (bron vermelden). Voor de code is geen licentie vastgelegd.
- **Citeren:** DataLab I, groep 4 (2026). *Voorspelmodel elektriciteitslevering per woning*, versie neighbourhood_electricity_delivery-20261110. Gegevens: CBS 85618NED - Kerncijfers wijken en buurten 2023.

## Wat de schatting meet

Gemiddelde jaarlijkse elektriciteitslevering uit het openbare net op individuele aansluitingen van particuliere woningen, alle woningtypen samen. Collectieve levering (bijv. lift, galerijverlichting) telt niet mee. Direct verbruikte eigen opwek (zonnepanelen) zit hier niet volledig in.

- **Eenheid:** kWh per woning per jaar
- **Jaar:** 2023
- **Niveau:** CBS-buurt (wijk- en buurtindeling 2023)

Wat de schatting níet is:
- Het volledige elektriciteitsverbruik achter de meter (eigen opwek telt niet volledig mee).
- Het totale energiegebruik van een buurt.
- Het verbruik van één huishouden.
- Het piekvermogen of de kans op netcongestie.

## Bedoeld gebruik

- Een schatting van de **gemiddelde** jaarlijkse elektriciteitslevering uit het net per particuliere woning in een Nederlandse buurt in 2023, op basis van zes buurtkenmerken.
- Verkennen hoe buurtkenmerken samenhangen met die levering, en buurten en gemeenten vergelijken.
- Onderwijs: laten zien hoe je een voorspelmodel eerlijk test en uitlegt.

## Niet bedoeld voor

- De energierekening of het verbruik van **één huishouden** (het is een buurtgemiddelde).
- Andere jaren dan 2023, of buurten buiten Nederland.
- Piekvermogen, netcongestie of netplanning.
- Beslissingen over personen of huishoudens, of beleid zonder eigen controle van de cijfers.
- Buurten met heel weinig woningen: daar publiceert het CBS meestal geen cijfer en is het model niet getest.

## Gegevens

- **Bron:** CBS 85618NED - Kerncijfers wijken en buurten 2023 (CBS-buurtcijfers 2023).
- **Training:** 10.878 buurten in 273 gemeenten. **Test:** 2.575 buurten in 69 andere gemeenten (indeling per gemeente, zodat buurten uit dezelfde gemeente niet in training én test zitten).
- Buurten zonder gepubliceerd CBS-cijfer doen niet mee aan training of test.
- **Kenmerken:**
  - Aandeel meergezinswoningen (flats, appartementen) – %
  - Aandeel woningen gebouwd in 2000 of later – %
  - Gemiddelde huishoudensgrootte – personen per huishouden
  - Stedelijkheid (CBS-klasse) – 1 = zeer sterk stedelijk ... 5 = niet stedelijk
  - Gemiddelde WOZ-waarde van woningen – x 1.000 euro
  - Aandeel koopwoningen – %

## Evaluatie (testgemeenten)

- **MAE** 270 kWh · **RMSE** 380 kWh · **R²** 0,76
- Ter vergelijking, steeds de mediaan voorspellen: MAE 660 kWh (58,7% minder fout).
- **MAE per groep:**
  - stedelijkheid 1 · zeer sterk: 170 kWh (238 buurten)
  - stedelijkheid 2 · sterk: 200 kWh (385 buurten)
  - stedelijkheid 3 · matig: 220 kWh (348 buurten)
  - stedelijkheid 4 · weinig: 300 kWh (416 buurten)
  - stedelijkheid 5 · niet: 320 kWh (1.188 buurten)
  - alle kenmerken bekend: 250 kWh (2.423 buurten)
  - minstens één kenmerk ontbreekt: 570 kWh (152 buurten)
- Het model trekt uitschieters naar het midden: buurten met een heel hoge levering worden te laag geschat, buurten met een heel lage levering te hoog (zie de tab Model).

## Uitleg van het model

- **Per schatting:** de app toont met SHAP-waarden hoeveel elk kenmerk de schatting omhoog of omlaag duwt ten opzichte van de gemiddelde schatting (SHAP: Lundberg & Lee, 2017; berekend met TreeExplainer: Lundberg et al., 2020). Bij een leeg veld is dat het effect van 'onbekend'.
- **Belang over alle testbuurten** (hoeveel stijgt de MAE als je een kenmerk door elkaar husselt): Gemiddelde WOZ-waarde van woningen +150 kWh; Stedelijkheid (CBS-klasse) +124 kWh; Aandeel koopwoningen +68 kWh; Gemiddelde huishoudensgrootte +48 kWh; Aandeel meergezinswoningen (flats, appartementen) +17 kWh; Aandeel woningen gebouwd in 2000 of later +8 kWh. Meergezinswoningen, koopwoningen, huishoudensgrootte en stedelijkheid hangen sterk samen (rangcorrelatie 0,6 à 0,7 in de training) en kunnen elkaar deels vervangen; hun belang wordt zo te laag ingeschat.
- Dit verklaart het **model**, niet de werkelijkheid: het zijn samenhangen, geen oorzaken.

## Beperkingen en risico's

- **Zonnepanelen** ontbreken als kenmerk, terwijl ze de levering uit het net direct verlagen.
- **Samenhang, geen oorzaak.** WOZ-waarde en koopwoningen zijn indirecte aanwijzingen voor woninggrootte en huishoudtype; ze hangen ook onderling samen.
- **Ontbrekende kenmerken** geven een veel grotere fout (zie evaluatie).
- **Afronding.** CBS rondt energiecijfers af op tientallen kWh.
- **Geen betrouwbaarheidsinterval per schatting.** Dat is niet onderzocht en wordt daarom niet getoond; de test-MAE is een gemiddelde, geen garantie.
- **Ongewone combinaties** worden gemarkeerd als alle kenmerken bekend zijn (Local Outlier Factor, grens bij de 1% meest afwijkende trainingsbuurten). In de testgemeenten gebeurde dit maar bij 25 van de 2.423 buurten met alle kenmerken bekend: te weinig om de fout betrouwbaar te meten, al was die in deze kleine groep groter dan bij de andere buurten.

_Opbouw volgens Mitchell et al. (2019), Model Cards for Model Reporting. Gegenereerd door src/explain.py uit de modelbestanden._
