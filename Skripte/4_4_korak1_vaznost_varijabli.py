"""
Poglavlje 4.4. Analiza osjetljivosti modela (važnost varijabli)
Korak 1: Izračun i vizualizacija važnosti varijabli (Mean Decrease Gini)
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

with open('rf_results.pkl','rb') as f:
    r = pickle.load(f)
with open('train_test_split.pkl','rb') as f:
    d = pickle.load(f)

rf_A = r['rf_A']
rf_B = r['rf_B']

feat_A = pd.Series(rf_A.feature_importances_, index=[c.strip() for c in d['X_A_train'].columns])
feat_A = feat_A.sort_values(ascending=False)
print("=== Važnost varijabli - Model A ===")
print(feat_A.round(3))

feat_B = pd.Series(rf_B.feature_importances_, index=d['X_B_train'].columns)
feat_B = feat_B.sort_values(ascending=False)
print("\n=== Važnost varijabli - Model B ===")
print(feat_B.round(3))

feat_A.to_pickle('importance_A.pkl')
feat_B.to_pickle('importance_B.pkl')

# Graf: Slika 10 - Važnost varijabli za oba modela
fig, axes = plt.subplots(1,2, figsize=(13,5.5))

axes[0].barh(feat_A.index[::-1], feat_A.values[::-1], color='#2e86c1')
axes[0].set_title('Model A: Važnost 12 komponenti indeksa')
axes[0].set_xlabel('Važnost varijable (Mean Decrease Gini)')

axes[1].barh(feat_B.index[::-1], feat_B.values[::-1], color='#e67e22')
axes[1].set_title('Model B: Važnost kontekstualnih varijabli')
axes[1].set_xlabel('Važnost varijable (Mean Decrease Gini)')

plt.suptitle('Analiza važnosti varijabli - Slučajna šuma', fontsize=13)
plt.tight_layout()
plt.savefig('slika10_vaznost_varijabli.png', dpi=150)
