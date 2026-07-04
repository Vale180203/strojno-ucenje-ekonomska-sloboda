"""
Poglavlje 4.2. Izrada klasifikacijskih modela za predviđanje ekonomske slobode
Korak 2: Treniranje stabla odlučivanja (Model A i Model B)
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with open('train_test_split.pkl','rb') as f:
    d = pickle.load(f)

X_A_train, X_A_test = d['X_A_train'], d['X_A_test']
X_B_train, X_B_test = d['X_B_train'], d['X_B_test']
y_train, y_test = d['y_train'], d['y_test']

# Stablo odlučivanja - Model A (12 komponenti)
tree_A = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
tree_A.fit(X_A_train, y_train)
pred_A = tree_A.predict(X_A_test)
acc_A = accuracy_score(y_test, pred_A)

print("=== STABLO ODLUČIVANJA - MODEL A (12 komponenti) ===")
print(f"Točnost na testnom skupu: {acc_A*100:.1f}%")
print("Matrica konfuzije:\n", confusion_matrix(y_test, pred_A))
print(classification_report(y_test, pred_A, target_names=['Niža razina','Viša razina']))

# Stablo odlučivanja - Model B (kontekstualne varijable)
tree_B = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
tree_B.fit(X_B_train, y_train)
pred_B = tree_B.predict(X_B_test)
acc_B = accuracy_score(y_test, pred_B)

print("\n=== STABLO ODLUČIVANJA - MODEL B (kontekstualne varijable) ===")
print(f"Točnost na testnom skupu: {acc_B*100:.1f}%")
print("Matrica konfuzije:\n", confusion_matrix(y_test, pred_B))
print(classification_report(y_test, pred_B, target_names=['Niža razina','Viša razina']))

# Graf: Slika 6 - Matrice konfuzije za oba modela
labels = ['Niža razina','Viša razina']
fig, axes = plt.subplots(1,2, figsize=(11,4.5))
for ax, pred, title, acc in zip(axes, [pred_A, pred_B],
                                   ['Model A (12 komponenti)','Model B (kontekstualne varijable)'],
                                   [acc_A, acc_B]):
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_xticklabels(labels)
    ax.set_yticks([0,1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predviđena klasa')
    ax.set_ylabel('Stvarna klasa')
    ax.set_title(f"{title}\nTočnost = {acc*100:.1f}%")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]), ha='center', va='center', fontsize=14,
                     color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.suptitle('Matrice konfuzije - Stablo odlučivanja', fontsize=13)
plt.tight_layout()
plt.savefig('slika6_matrice_konfuzije_stablo.png', dpi=150)

with open('tree_results.pkl','wb') as f:
    pickle.dump({'tree_A':tree_A,'tree_B':tree_B,'pred_A':pred_A,'pred_B':pred_B,
                 'acc_A':acc_A,'acc_B':acc_B}, f)
