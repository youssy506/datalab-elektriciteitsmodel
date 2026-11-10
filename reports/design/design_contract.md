# Ontwerpcontract: app elektriciteitslevering per woning

## Product en gebruiker
- **Oppervlak:** lokale Streamlit-app bij het DataLab-model (één opgeslagen pipeline).
- **Gebruikers:** de projectgroep, docenten/beoordelaars en een Enexis-analist als denkbeeldige opdrachtgever.
  Ze gebruiken de app om (1) een schatting te maken, (2) een echte buurt te controleren,
  (3) gemeenten te vergelijken en (4) te zien hoe betrouwbaar het model is.
- **Primaire actie:** kenmerken invullen → "Schatting berekenen" → getal + context lezen.
- **Wat eerst gescand wordt:** het geschatte getal met eenheid, daarna "is dat hoog of laag?", daarna "hoe zeker?".

## Richting
- **Toon:** rustig, technisch-redactioneel ("meetrapport"), geen marketing. Licht thema, warm papierwit,
  koppen in serif (Source Serif, ingebouwd in Streamlit) en tekst in sans.
- **Kleur:** diep petrol (#0F5F5C) voor acties, het model en CBS-waarden; amber alleen voor
  "jouw schatting / deze gemeente" (#B07600 voor vormen, 3,6:1; #8A5A00 voor tekst, 5,6:1);
  grijs-blauw (#5B7083) voor de referentie (alle buurten). Toelichtingen in #4F5866 (6,8:1) in plaats van
  st.caption (60% transparant, ca. 4,1:1). Badges en meldingen met eigen themakleuren (alle ≥ 4,5:1).
  Geen paars, geen gradients, geen Material-iconen (hun ligatuurtekst lekt naar schermlezers).
- **Herinneringsdetail:** de *leveringsmeter*: een horizontale band die laat zien waar de schatting
  valt tussen de 10.878 trainingsbuurten (P10–P90-band, mediaanstreep, amber markering) plus
  "hoger dan X% van de buurten". Dit is een verdeling van buurten, **geen** onzekerheidsinterval.

## Constraints
- Streamlit 1.58; Altair zit al in Streamlit. Eén bewuste nieuwe dependency: `shap` (uitleg per schatting).
- Netwerk alleen voor de postcodezoeker (PDOK Locatieserver); zonder internet werkt de rest gewoon.
- Eén voorspelroute: `src/predict.py` (app en batch gelijk).
- Nederlandse labels en getalnotatie (1.234,5).
- Geen betrouwbaarheidsinterval tonen; test-MAE niet als garantie presenteren.
- Toegankelijkheid: contrast ≥ 4.5:1 voor tekst, labels bij elk veld, toetsenbord bruikbaar.

## Doelviewports
- `1440x900` (desktop): invoer links, resultaat rechts, zonder scrollen zichtbaar.
- `390x844` (mobiel): kolommen gestapeld, geen horizontale scroll, tekst breekt netjes af.

## Hiërarchie per tab
| Tab | Eerste blik | Daarna |
|---|---|---|
| Schatting | Getal (kWh/woning/jaar) | Meter (+ eigen levering), nauwkeurigheid, waarschuwingen, 'Wat als', 'Waarom' (SHAP), download; CSV-upload onderaan |
| Buurt | Postcode → CBS-waarde vs schatting | Train/test-status, verschil, eigen levering, 'Waarom', vergelijking met gemeente/NL/tweede buurt |
| Gemeenten | Waarde + rang van gekozen gemeente | Spreiding van haar buurten, ranglijst, download |
| Model | Test-MAE vs baseline | Scatter, modelvergelijking, subgroepen, kenmerkbelang, combinatiecontrole, download alle buurten |
| Uitleg | Modelkaart (Mitchell et al., 2019) | Wat het meet / niet is, bedoeld gebruik, gegevens, evaluatie, beperkingen, download |

## Invoer
- Zes velden; leeg = onbekend (het model gebruikt dan ook een 'ontbreekt'-indicator, dus het is níet hetzelfde
  als de mediaan invullen). Stedelijkheid is een selectbox met 'onbekend' (toegankelijk label en waarde).
- Helemaal leeg verzenden = foutstatus (ook per rij in een CSV). Voorbeeld kiezen of leegmaken wist het oude resultaat.
- Eigen levering: optioneel, 100–25.000 kWh, als donker vierkant op de meter; altijd met de uitleg dat één
  huishouden geen buurtgemiddelde is.
- Ongewone combinatie (alleen als alle zes kenmerken bekend zijn): waarschuwing met wat de test erover zegt.

## Inhoudsstatussen
| Status | Trigger | Zichtbare feedback | Veilige volgende stap |
|---|---|---|---|
| Laden | Eerste start (model en tabellen in cache) | Spinner "Model en gegevens laden…" | Wachten |
| Leeg | Nog geen schatting gemaakt | Resultaatpaneel met uitleg en verwijzing naar voorbeelden | Voorbeeld kiezen of velden invullen |
| Fout | Ongeldige invoer, helemaal lege invoer, ontbrekende of beschadigde modelbestanden | Rode melding met precies welk veld/bestand | Veld aanpassen / `python src/train.py` draaien |
| Gedeeltelijk | Leeg veld, buurt zonder CBS-waarde of kenmerken, buurt zonder woningen, gemeente zonder cijfer | Schatting met lijst onbekende velden + één waarschuwing; "niet gepubliceerd"; "geen woningen" (geen schatting) | Ontbrekende waarde invullen / andere buurt kiezen |
| Succes | Geldige invoer | Getal, meter, nauwkeurigheid | Andere waarden proberen / buurt vergelijken |
| Toegang | n.v.t. | Lokale app zonder accounts; alleen openbare CBS-data | — |
