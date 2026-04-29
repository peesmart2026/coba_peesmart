import streamlit as st
import pickle
import numpy as np

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Deteksi Kesehatan Urine",
    layout="centered"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    with open('model_peesmart.sav', 'rb') as file:
        return pickle.load(file)

def main():
    st.title("🧪 Sistem Deteksi Kesehatan Urine")
    st.write(
        "Masukkan nilai pH dan RGB urine untuk mendeteksi "
        "dehidrasi, diabetes mellitus, dan gangguan fungsi ginjal."
    )

    try:
        # Load file model
        data_load = load_model()

        # Ambil model
        scaler = data_load['scaler']
        knn_model = data_load['knn_dehidrasi']
        nb_model = data_load['naive_bayes_diabetes']
        svm_model = data_load['svm_gangguan_ginjal']

        # =========================
        # Form Input
        # =========================
        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                ph = st.number_input(
                    "Nilai pH",
                    min_value=0.0,
                    max_value=14.0,
                    value=6.0,
                    step=0.1
                )

                r = st.number_input(
                    "Nilai Red (R)",
                    min_value=0,
                    max_value=255,
                    value=255
                )

            with col2:
                g = st.number_input(
                    "Nilai Green (G)",
                    min_value=0,
                    max_value=255,
                    value=255
                )

                b = st.number_input(
                    "Nilai Blue (B)",
                    min_value=0,
                    max_value=255,
                    value=0
                )

            submit = st.form_submit_button("Prediksi")

        # =========================
        # Prediksi
        # =========================
        if submit:
            input_data = np.array([[ph, r, g, b]])

            # Scaling
            input_scaled = scaler.transform(input_data)

            # Prediksi masing-masing model
            pred_knn = knn_model.predict(input_scaled)[0]
            pred_nb = nb_model.predict(input_scaled)[0]
            pred_svm = svm_model.predict(input_scaled)[0]

            # =========================
            # Hasil
            # =========================
            st.divider()
            st.subheader("Hasil Analisis")

            # Dehidrasi
            if pred_knn == 1:
                st.warning("⚠️ Dehidrasi Terdeteksi")
            else:
                st.success("✅ Tidak Dehidrasi")

            # Diabetes
            if pred_nb == 1:
                st.warning("⚠️ Diabetes Mellitus Terdeteksi")
            else:
                st.success("✅ Tidak Diabetes Mellitus")

            # Gangguan Ginjal
            if pred_svm == 1:
                st.error("🚨 Gangguan Fungsi Ginjal Terdeteksi")
            else:
                st.success("✅ Tidak Ada Gangguan Fungsi Ginjal")

    except FileNotFoundError:
        st.error("File 'model_peesmart.sav' tidak ditemukan.")

    except KeyError as e:
        st.error(f"Key model tidak ditemukan: {e}")

    except Exception as e:
        st.error(f"Terjadi error: {e}")

if __name__ == "__main__":
    main()