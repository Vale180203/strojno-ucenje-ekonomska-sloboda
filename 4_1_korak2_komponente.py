"""
Poglavlje 4.1. Eksploratorna analiza podataka
Korak 2: Deskriptivna statistika i distribucija 12 komponenti indeksa (Model A)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Učitavanje podataka (nastavak iz koraka 1)
df = pd.read_pickle('data_with_target.pkl')
# Napomena: ako pokrećeš ovaj korak zasebno bez prethodnog koraka, umjesto gornje linije koristi:
# df = pd.read_excel('economic_freedom_2019_cleaned.xlsx', sheet_name='Data')

comps = ['Property Rights','Judical Effectiveness','Government Integrity','Tax Burden',
         "Gov't Spending",'Fiscal Health','Business Freedom','Labor Freedom',
         'Monetary Freedom','Trade Freedom','Investment Freedom ','Financial Freedom']

# Deskriptivna statistika
stats = df[comps].describe().T[['mean','std','min','max']].round(1)
stats.columns = ['Prosjek (M)','Std. devijacija (SD)','Min','Max']
stats = stats.sort_values('Prosjek (M)', ascending=False)
print(stats)

# Graf: Slika 2 - Box-plot distribucije 12 komponenti
order = df[comps].mean().sort_values(ascending=True).index

fig, ax = plt.subplots(figsize=(9,6))
box = ax.boxplot([df[c].dropna() for c in order], orientation='horizontal', patch_artist=True,
                  tick_labels=[c.strip() for c in order])

for patch in box['boxes']:
    patch.set_facecolor('#5dade2')
    patch.set_alpha(0.7)

ax.set_xlabel('Ocjena (0-100)')
ax.set_title('Distribucija 12 komponenti Indeksa ekonomske slobode (N=180)')
ax.set_xlim(-5,105)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('slika2_distribucija_komponenti.png', dpi=150)