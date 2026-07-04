"""
Poglavlje 4.1. Eksploratorna analiza podataka
Korak 3: Korelacijska matrica 12 komponenti indeksa (Model A)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import numpy as np

df = pd.read_pickle('data_with_target.pkl')
# Ako pokrećeš zasebno bez prethodnih koraka:
# df = pd.read_excel('economic_freedom_2019_cleaned.xlsx', sheet_name='Data')

comps = ['Property Rights','Judical Effectiveness','Government Integrity','Tax Burden',
         "Gov't Spending",'Fiscal Health','Business Freedom','Labor Freedom',
         'Monetary Freedom','Trade Freedom','Investment Freedom ','Financial Freedom']
labels = [c.strip() for c in comps]

corr = df[comps].corr()
corr.columns = labels
corr.index = labels

# Ispis 5 najjačih korelacija
c = pd.DataFrame(np.where(np.eye(len(labels)), np.nan, corr.values), index=labels, columns=labels)
pairs = c.unstack().dropna().sort_values(ascending=False)
print("Pet najjačih pozitivnih korelacija:")
seen = set()
count = 0
for (a,b), v in pairs.items():
    key = frozenset([a,b])
    if key in seen: continue
    seen.add(key)
    print(f"  {a} <-> {b}: r = {v:.2f}")
    count += 1
    if count == 5: break

# Graf: Slika 3 - Korelacijska matrica (toplinska mapa)
fig, ax = plt.subplots(figsize=(9,7.5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j,i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=7,
                color='white' if abs(corr.iloc[i,j])>0.6 else 'black')
plt.colorbar(im, ax=ax, label='Pearsonov koeficijent korelacije (r)')
ax.set_title('Korelacijska matrica 12 komponenti Indeksa ekonomske slobode')
plt.tight_layout()
plt.savefig('slika3_korelacijska_matrica.png', dpi=150)
