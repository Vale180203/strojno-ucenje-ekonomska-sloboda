# Strojno učenje u analizi ekonomske slobode

Empirijski dio diplomskog rada **"Strojno učenje u analizi ekonomske slobode: klasifikacija zemalja i identifikacija ključnih pokazatelja"**, izrađenog na Ekonomskom fakultetu u Osijeku (Sveučilišni diplomski studij Poslovna ekonomija, smjer Poslovna informatika).

**Autor:** Valentin Pavlović
**Mentor:** doc. dr. sc. Domagoj Ševerdija
**Komentor:** dr. sc. Adela Has

## O radu

Rad istražuje primjenu stabala odlučivanja i slučajnih šuma u klasifikaciji zemalja prema razini ekonomske slobode, na temelju Indeksa ekonomske slobode zaklade Heritage Foundation (2019. izdanje). Izgrađena su dva modela:
- **Model A** — temeljen na 12 komponenti indeksa
- **Model B** — temeljen isključivo na kontekstualnim makroekonomskim pokazateljima (BDP po glavi stanovnika, nezaposlenost, inflacija i dr.)

## Izvor podataka

Duncan, L. (2019). *The Economic Freedom Index* [skup podataka]. Kaggle.
Dostupno na: https://www.kaggle.com/datasets/lewisduncan93/the-economic-freedom-index

Očišćena verzija koja se koristi u analizi nalazi se u `economic_freedom_2019_cleaned.xlsx`.

## Struktura repozitorija

| Mapa/Datoteka | Opis |
|---|---|
| `Skripte/` | Sve Python skripte (poglavlje 4.) |
| `Slike/` | Grafovi generirani skriptama, korišteni u radu |
| `economic_freedom_2019_cleaned.xlsx` | Očišćen skup podataka |
| `*.pkl` | Međuspremnici podataka između koraka (nisu potrebni za pokretanje ispočetka) |

## Pokretanje

Skripte je potrebno pokrenuti iz **glavne mape** repozitorija (ne iz `Skripte/`), navedenim redoslijedom:

```bash
pip install pandas matplotlib scikit-learn openpyxl
python Skripte/4_1_korak1_ciljna_varijabla.py
python Skripte/4_1_korak2_komponente.py
python Skripte/4_1_korak3_korelacije.py
python Skripte/4_1_korak4_outlieri.py
python Skripte/4_1_korak5_regije.py
python Skripte/4_2_korak1_podjela_podataka.py
python Skripte/4_2_korak2_stablo_odlucivanja.py
python Skripte/4_2_korak3_vizualizacija_stabla.py
python Skripte/4_2_korak4_slucajna_suma.py
python Skripte/4_3_korak1_usporedba.py
python Skripte/4_4_korak1_vaznost_varijabli.py
```