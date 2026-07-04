"""
Poglavlje 4.1. Eksploratorna analiza podataka
Korak 4: Provjera outliera (stopa inflacije)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_pickle('data_with_target.pkl')
# Ako pokrećeš zasebno: df = pd.read_excel('economic_freedom_2019_cleaned.xlsx', sheet_name='Data')

# Provjera ekstremnih vrijednosti (Z-score > 3) kroz makroekonomske varijable
macro = ['Population (Millions)','GDP (Billions, PPP)','GDP per Capita (PPP)',
          'Unemployment (%)','Inflation (%)','FDI Inflow (Millions)','Public Debt (% of GDP)']
for col in macro:
    z = (df[col] - df[col].mean()) / df[col].std()
    outliers = df.loc[z.abs() > 3, ['Country Name', col]]
    if len(outliers) > 0:
        print(f"\n{col}:")
        print(outliers.to_string(index=False))

# Graf: Slika 4 - Top 10 zemalja po inflaciji (naglašen ekstremni outlier)
top10 = df.nlargest(10, 'Inflation (%)')[['Country Name','Inflation (%)']].sort_values('Inflation (%)')
fig, ax = plt.subplots(figsize=(8,5))
colors = ['#c0392b' if v > 500 else '#5dade2' for v in top10['Inflation (%)']]
ax.barh(top10['Country Name'], top10['Inflation (%)'], color=colors)
ax.set_xlabel('Stopa inflacije (%)')
ax.set_title('10 zemalja s najvišom stopom inflacije (2019)')
for i,(v,n) in enumerate(zip(top10['Inflation (%)'], top10['Country Name'])):
    ax.text(v+15, i, f"{v:.1f}%", va='center', fontsize=9)
plt.tight_layout()
plt.savefig('slika4_outlier_inflacija.png', dpi=150)
