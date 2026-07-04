"""
Poglavlje 4.1. Eksploratorna analiza podataka
Korak 1: Konstrukcija ciljne varijable i vizualizacija distribucije
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Učitavanje očišćenog skupa podataka
df = pd.read_excel('economic_freedom_2019_cleaned.xlsx', sheet_name='Data')

# Konstrukcija ciljne varijable (prag 60 bodova, dogovoreno u poglavlju 3.2.)
df['Ciljna_varijabla'] = (df['2019 Score'] >= 60).astype(int)
df['Ciljna_varijabla_naziv'] = df['Ciljna_varijabla'].map({
    0: 'Niža razina slobode',
    1: 'Viša razina slobode'
})

print(f"Broj zemalja: {len(df)}")
print(df['Ciljna_varijabla_naziv'].value_counts())
print((df['Ciljna_varijabla_naziv'].value_counts(normalize=True)*100).round(1))

# Graf: Slika 1 - Distribucija ciljne varijable
fig, ax = plt.subplots(figsize=(6,4.5))
counts = df['Ciljna_varijabla_naziv'].value_counts().reindex(
    ['Niža razina slobode','Viša razina slobode'])
bars = ax.bar(counts.index, counts.values, color=['#c0392b','#27ae60'], width=0.5)

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, val+1.5, str(val),
            ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Broj zemalja')
ax.set_title('Distribucija ciljne varijable (N=180)')
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig('slika1_distribucija_ciljne.png', dpi=150)

# Spremanje podataka za sljedeće korake
df.to_pickle('data_with_target.pkl')
