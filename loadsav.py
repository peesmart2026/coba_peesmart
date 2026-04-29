import pickle
import numpy as np

# =========================
# Muat file model
# =========================
with open('model_peesmart.sav', 'rb') as file:
    data_load = pickle.load(file)

# =========================
# Ambil komponennya
# =========================
scaler = data_load['scaler']
knn_model = data_load['knn_dehidrasi']
nb_model = data_load['naive_bayes_diabetes']
svm_model = data_load['svm_gangguan_ginjal']

# =========================
# Contoh Data Baru
# Format:
# [pH, R, G, B]
# =========================
data_baru = np.array([
    [6.0, 255.0, 255.0, 0.0]
])

# =========================
# Scaling Data
# =========================
data_scaled = scaler.transform(data_baru)

# =========================
# Prediksi per Model
# =========================
hasil_knn = knn_model.predict(data_scaled)
hasil_nb = nb_model.predict(data_scaled)
hasil_svm = svm_model.predict(data_scaled)

# =========================
# Tampilkan Hasil
# =========================
print("=== HASIL PREDIKSI ===")

if hasil_knn[0] == 1:
    print("KNN: Dehidrasi Terdeteksi")
else:
    print("KNN: Tidak Dehidrasi")

if hasil_nb[0] == 1:
    print("Naive Bayes: Diabetes Mellitus Terdeteksi")
else:
    print("Naive Bayes: Tidak Diabetes Mellitus")

if hasil_svm[0] == 1:
    print("SVM: Gangguan Fungsi Ginjal Terdeteksi")
else:
    print("SVM: Tidak Gangguan Fungsi Ginjal")