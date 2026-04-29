import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================
# 1. Memuat Dataset
# =========================
df = pd.read_csv('dataset_urine.csv')

# =========================
# 2. Feature
# =========================
X = df[['pH', 'R', 'G', 'B']]

# =========================
# 3. Target per Penyakit
# Binary Classification
# =========================
y_dehidrasi = (df['Status'] == 'Dehidrasi').astype(int)
y_diabetes = (df['Status'] == 'Diabetes Mellitus').astype(int)
y_ginjal = (df['Status'] == 'Gangguan Fungsi Ginjal').astype(int)

# =========================
# Function Training + Evaluasi
# =========================
def train_and_evaluate(model, X, y, model_name):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training
    model.fit(X_train_scaled, y_train)

    # Prediksi
    y_pred = model.predict(X_test_scaled)

    # Evaluasi
    print(f"\n===== {model_name} =====")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Tidak', 'Ya']
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Tidak', 'Ya'],
        yticklabels=['Tidak', 'Ya']
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# =========================
# 4. Model KNN (Dehidrasi)
# =========================
knn_model = KNeighborsClassifier(n_neighbors=3)
train_and_evaluate(
    knn_model,
    X,
    y_dehidrasi,
    "KNN - Dehidrasi"
)

# =========================
# 5. Model Naive Bayes (Diabetes)
# =========================
nb_model = GaussianNB()
train_and_evaluate(
    nb_model,
    X,
    y_diabetes,
    "Naive Bayes - Diabetes Mellitus"
)

# =========================
# 6. Model SVM (Gangguan Ginjal)
# =========================
svm_model = SVC(
    kernel='linear',
    random_state=42
)

train_and_evaluate(
    svm_model,
    X,
    y_ginjal,
    "SVM - Gangguan Fungsi Ginjal"
)