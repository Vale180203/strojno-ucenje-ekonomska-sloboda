"""
Poglavlje 4.3. Usporedba uspješnosti modela
Korak 1: Objedinjena tablica i graf usporedbe svih modela
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import numpy as np

with open('tree_results.pkl','rb') as f:
    tree_r = pickle.load(f)
with open('rf_results.pkl','rb') as f:
    rf_r = pickle.load(f)

usporedba = pd.DataFrame({
    'Model': ['Model A (12 komponenti)', 'Model A (12 komponenti)',
              'Model B (kontekstualne)', 'Model B (kontekstualne)'],
    'Algoritam': ['Stablo odlučivanja', 'Slučajna šuma', 'Stablo odlučivanja', 'Slučajna šuma'],
    'Tocnost (%)': [tree_r['acc_A']*100, rf_r['acc_A']*100, tree_r['acc_B']*100, rf_r['acc_B']*100],
    'OOB tocnost (%)': [None, rf_r['oob_A']*100, None, rf_r['oob_B']*100]
})
print(usporedba.round(1))
usporedba.to_pickle('usporedba_modela.pkl')

# Graf: Slika 9 - Usporedba točnosti svih modela
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(2)
width = 0.35

modelA = usporedba[usporedba['Model']=='Model A (12 komponenti)']['Tocnost (%)'].values
modelB = usporedba[usporedba['Model']=='Model B (kontekstualne)']['Tocnost (%)'].values

bars1 = ax.bar(x - width/2, modelA, width, label='Model A (12 komponenti)', color='#2e86c1')
bars2 = ax.bar(x + width/2, modelB, width, label='Model B (kontekstualne varijable)', color='#e67e22')

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%", ha='center', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(['Stablo odlučivanja','Slučajna šuma'])
ax.set_ylabel('Točnost na testnom skupu (%)')
ax.set_title('Usporedba točnosti klasifikacijskih modela')
ax.set_ylim(0,105)
ax.legend()
plt.tight_layout()
plt.savefig('slika9_usporedba_modela.png', dpi=150)
