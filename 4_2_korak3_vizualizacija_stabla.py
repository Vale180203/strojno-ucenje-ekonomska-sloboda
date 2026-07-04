"""
Poglavlje 4.2. Izrada klasifikacijskih modela za predviđanje ekonomske slobode
Korak 3: Vizualizacija strukture stabla odlučivanja (Model A)
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.tree import plot_tree, export_text

with open('tree_results.pkl','rb') as f:
    r = pickle.load(f)
with open('train_test_split.pkl','rb') as f:
    d = pickle.load(f)

tree_A = r['tree_A']
feature_names_A = [c.strip() for c in d['X_A_train'].columns]

# Graf: Slika 7 - Struktura stabla odlučivanja
fig, ax = plt.subplots(figsize=(20,10))
plot_tree(tree_A, feature_names=feature_names_A, class_names=['Niža razina','Viša razina'],
          filled=True, rounded=True, fontsize=9, ax=ax)
plt.title('Stablo odlučivanja - Model A (12 komponenti), max_depth=4', fontsize=14)
plt.tight_layout()
plt.savefig('slika7_stablo_vizualizacija.png', dpi=130)

# Tekstualni prikaz pravila (koristan za opis u tekstu rada)
print("=== Pravila stabla (tekstualni prikaz) ===")
print(export_text(tree_A, feature_names=feature_names_A))
