import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv('dataset_urine.csv')

# =========================
# 2. Ambil Feature
# =========================
X = df[['pH', 'R', 'G', 'B']]

# =========================
# 3. Buat Target per Penyakit
# (Binary Classification)
# =========================
y_dehidrasi = (df['Status'] == 'Dehidrasi').astype(int)
y_diabetes = (df['Status'] == 'Diabetes Mellitus').astype(int)
y_ginjal = (df['Status'] == 'Gangguan Fungsi Ginjal').astype(int)

# =========================
# 4. Scaling Data
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Model KNN (Dehidrasi)
# =========================
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_scaled, y_dehidrasi)

# =========================
# 6. Model Naive Bayes (Diabetes Mellitus)
# =========================
nb_model = GaussianNB()
nb_model.fit(X_scaled, y_diabetes)

# =========================
# 7. Model SVM (Gangguan Fungsi Ginjal)
# =========================
svm_model = SVC(
    kernel='linear',
    probability=True,
    random_state=42
)
svm_model.fit(X_scaled, y_ginjal)

# =========================
# 8. Simpan Semua Model
# =========================
model_data = {
    'scaler': scaler,
    'knn_dehidrasi': knn_model,
    'naive_bayes_diabetes': nb_model,
    'svm_gangguan_ginjal': svm_model
}

# Simpan ke file .sav
with open('model_peesmart.sav', 'wb') as file:
    pickle.dump(model_data, file)

print("Berhasil! Semua model disimpan dalam 'model_peesmart.sav'")