"""
Poglavlje 4.2. Izrada klasifikacijskih modela za predviđanje ekonomske slobode
Korak 1: Priprema konačnog skupa (ciljna varijabla + imputacija) i podjela na
skup za učenje i skup za testiranje.

Ova skripta je samostalna: radi izravno iz "economic_freedom_2019_cleaned.xlsx"
(rezultat poglavlja 3.3.), bez potrebe za bilo kojom drugom datotekom.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# 1) Učitavanje očišćenog skupa podataka (rezultat poglavlja 3.3.)
df = pd.read_excel('economic_freedom_2019_cleaned.xlsx', sheet_name='Data')

# 2) Konstrukcija ciljne varijable (prag 60 bodova, dogovoreno u 3.2.)
df['Ciljna_varijabla'] = (df['2019 Score'] >= 60).astype(int)

# 3) Imputacija nedostajućih vrijednosti medijanom regije (dogovoreno u 3.3.),
#    potrebna isključivo za varijable Modela B
macro = ['Population (Millions)','GDP (Billions, PPP)','GDP Growth Rate (%)',
          '5 Year GDP Growth Rate (%)','GDP per Capita (PPP)','Unemployment (%)',
          'Inflation (%)','FDI Inflow (Millions)','Public Debt (% of GDP)']
for col in macro:
    df[col] = df.groupby('Region')[col].transform(lambda x: x.fillna(x.median()))

# 4) Definicija ulaznih varijabli
comps = ['Property Rights','Judical Effectiveness','Government Integrity','Tax Burden',
         "Gov't Spending",'Fiscal Health','Business Freedom','Labor Freedom',
         'Monetary Freedom','Trade Freedom','Investment Freedom ','Financial Freedom']

macro_b = ['Population (Millions)','GDP (Billions, PPP)','GDP Growth Rate (%)',
           'GDP per Capita (PPP)','Unemployment (%)','Inflation (%)',
           'FDI Inflow (Millions)','Public Debt (% of GDP)']

X_A = df[comps]
# One-hot kodiranje regije: kategorička varijabla pretvara se u brojčane 0/1 stupce,
# jer algoritmi strojnog učenja zahtijevaju brojčani ulaz. drop_first=True izbjegava
# redundanciju (jedna regija ostaje referentna kategorija).
X_B = pd.get_dummies(df[macro_b + ['Region']], columns=['Region'], drop_first=True)
y = df['Ciljna_varijabla']

# 5) Podjela 80% učenje / 20% testiranje, uz stratifikaciju kako bi omjer klasa
#    ostao jednak u oba podskupa kao i u cjelokupnom uzorku
X_A_train, X_A_test, y_train, y_test = train_test_split(
    X_A, y, test_size=0.2, random_state=42, stratify=y)

X_B_train, X_B_test, y_train_b, y_test_b = train_test_split(
    X_B, y, test_size=0.2, random_state=42, stratify=y)

print("=== Model A (12 komponenti) ===")
print(f"Skup za učenje: {X_A_train.shape[0]} zemalja, Skup za testiranje: {X_A_test.shape[0]} zemalja")
print("Distribucija klasa u skupu za učenje:", y_train.value_counts().to_dict())
print("Distribucija klasa u skupu za testiranje:", y_test.value_counts().to_dict())

print("\n=== Model B (9 kontekstualnih varijabli, uz one-hot kodiranje regije) ===")
print(f"Skup za učenje: {X_B_train.shape[0]} zemalja, Skup za testiranje: {X_B_test.shape[0]} zemalja")
print(f"Broj varijabli nakon kodiranja regije: {X_B.shape[1]}")

# 6) Spremanje za sljedeće korake (4_2_korak2 itd. će ovo učitati)
with open('train_test_split.pkl','wb') as f:
    pickle.dump({
        'X_A_train':X_A_train,'X_A_test':X_A_test,
        'X_B_train':X_B_train,'X_B_test':X_B_test,
        'y_train':y_train,'y_test':y_test
    }, f)

print("\nPodjela spremljena u 'train_test_split.pkl' za sljedeći korak.")
