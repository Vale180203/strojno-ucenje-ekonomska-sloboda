"""
Poglavlje 4.1. Eksploratorna analiza podataka
Korak 5: Usporedba prosječne ekonomske slobode po regijama
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_pickle('data_with_target.pkl')
# Ako pokrećeš zasebno: df = pd.read_excel('economic_freedom_2019_cleaned.xlsx', sheet_name='Data')

regional = df.groupby('Region')['2019 Score'].agg(['mean','std','count']).round(1)
regional = regional.sort_values('mean', ascending=False)
regional.columns = ['Prosjek','Std. devijacija','Broj zemalja']
print(regional)

# Graf: Slika 5 - Prosječna razina ekonomske slobode po regijama
fig, ax = plt.subplots(figsize=(8,5))
order = regional.index
colors = plt.cm.RdYlGn([(v-40)/40 for v in regional['Prosjek']])
bars = ax.barh(order[::-1], regional['Prosjek'][::-1], color=colors[::-1])
for bar, val in zip(bars, regional['Prosjek'][::-1]):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va='center', fontsize=10)
ax.axvline(df['2019 Score'].mean(), color='black', linestyle='--', linewidth=1,
           label=f"Globalni prosjek ({df['2019 Score'].mean():.1f})")
ax.set_xlabel('Prosječna ocjena Indeksa ekonomske slobode (2019)')
ax.set_title('Prosječna razina ekonomske slobode po regijama')
ax.legend()
plt.tight_layout()
plt.savefig('slika5_regije.png', dpi=150)
