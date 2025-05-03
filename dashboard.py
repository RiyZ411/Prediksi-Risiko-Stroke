# Library
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import base64

# Fungsi untuk mendapatkan string base64 dari file gambar
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Fungsi untuk mengatur gambar lokal sebagai latar belakang halaman
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    opacity: 0.86;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)

# Menggunakan fungsi di atas
set_png_as_page_bg('./Images/stroke.jpeg')
#/home/riyan/Machine Learning Terapan/Proyek Pertama

# Memanggil best model
model = joblib.load("./best_model.joblib")

# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>Prediksi Stroke Berdasarkan Data Pasien</h1>", unsafe_allow_html=True)

# Sidebar contact
with st.sidebar:
    st.title("Profil")
    st.markdown("**Nama:** Riyan Zaenal Arifin")
    st.markdown("**Email:** riyanzaenal411@gmail.com")
    st.markdown("**Cohort ID:** A327YBF437")
    st.markdown("**Email Cohort:** A327YBF437@devacademy.id")

with st.form("predict"):
    # Input numerik
    age = st.number_input("Umur", min_value=0.0, max_value=82.0, step=1.0)
    
    hypertension_inp = st.selectbox("Riwayat Hipertensi", ["Ya", "Tidak"])
    hypertension = 1 if hypertension_inp == "Ya" else 0

    heart_disease_inp = st.selectbox("Riwayat Penyakit Jantung", ["Ya", "Tidak"])
    heart_disease = 1 if heart_disease_inp == "Ya" else 0

    avg_glucose_level = st.number_input("Rata-rata Kadar Gula Darah", min_value=55.0, max_value=271.0, step=1.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=97.0, step=1.0)
    
    # Input kategori: Gender
    gender_inp = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    if gender_inp == "Pria":
        gender_Female = 0
        gender_Male = 1
    else:
        gender_Female = 1
        gender_Male = 0

    # Input kategori: Pernah Menikah (ever married)
    ever_married_inp = st.selectbox("Pernah Menikah", ["Sudah Menikah", "Lajang"])
    if ever_married_inp == "Sudah Menikah":
        ever_married_No = 0
        ever_married_Yes = 1
    else:
        ever_married_No = 1
        ever_married_Yes = 0

    # Input kategori: Work Type
    work_type_inp = st.selectbox("Pekerjaan", ["Pemerintahan", "Tidak Bekerja", "Swasta", "Wiraswasta", "Anak-anak"])
    work_type_Govt_job = 1 if work_type_inp == "Pemerintahan" else 0
    work_type_Never_worked = 1 if work_type_inp == "Tidak Bekerja" else 0
    work_type_Private = 1 if work_type_inp == "Swasta" else 0
    work_type_Self_employed = 1 if work_type_inp == "Wiraswasta" else 0
    work_type_children = 1 if work_type_inp == "Anak-anak" else 0

    # Input kategori: Smoking Status
    smoking_status_inp = st.selectbox("Status Merokok", ["Pernah", "Tidak Merorok", "Merokok"])
    smoking_status_formerly_smoked = 1 if smoking_status_inp == "Pernah" else 0
    smoking_status_never_smoked = 1 if smoking_status_inp == "Tidak Merorok" else 0
    smoking_status_smokes = 1 if smoking_status_inp == "Merokok" else 0

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Bangun DataFrame input dengan urutan fitur yang sesuai
        input_data = pd.DataFrame([{
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "gender_Female": gender_Female,
            "gender_Male": gender_Male,
            "ever_married_No": ever_married_No,
            "ever_married_Yes": ever_married_Yes,
            "work_type_Govt_job": work_type_Govt_job,
            "work_type_Never_worked": work_type_Never_worked,
            "work_type_Private": work_type_Private,
            "work_type_Self-employed": work_type_Self_employed,
            "work_type_children": work_type_children,
            "smoking_status_formerly smoked": smoking_status_formerly_smoked,
            "smoking_status_never smoked": smoking_status_never_smoked,
            "smoking_status_smokes": smoking_status_smokes
        }])

        # Pastikan urutan kolom sesuai dengan data training (tanpa kolom target 'stroke')
        expected_columns = [
            "age",
            "hypertension",
            "heart_disease",
            "avg_glucose_level",
            "bmi",
            "gender_Female",
            "gender_Male",
            "ever_married_No",
            "ever_married_Yes",
            "work_type_Govt_job",
            "work_type_Never_worked",
            "work_type_Private",
            "work_type_Self-employed",
            "work_type_children",
            "smoking_status_formerly smoked",
            "smoking_status_never smoked",
            "smoking_status_smokes"
        ]
        input_data = input_data[expected_columns]

        # Normalisasi data dengan StandardScaler
        scaler = joblib.load('./scaler.joblib')
        data_normalized = scaler.transform(input_data)

        # Prediksi dengan model
        input_df = pd.DataFrame(data_normalized, columns=expected_columns)
        pred = model.predict(data_normalized)

        # Interpretasi hasil prediksi
        if pred[0] == 1:
            st.error("Beresiko Stroke ⚠️")
        else:
            st.success("Tidak Beresiko Stroke")