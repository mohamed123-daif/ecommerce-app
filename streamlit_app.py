import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import mysql.connector

# PDF Export
try:
    from fpdf import FPDF
    fpdf_installed = True
except ModuleNotFoundError:
    fpdf_installed = False

# Configuration principale (DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT)
st.set_page_config(page_title="Application de Prédiction des Ventes E-Commerce", layout="wide")

# Connexion à la base de données

def check_credentials(username, password):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="ecommerce"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        return cursor.fetchone() is not None
    except mysql.connector.Error as e:
        st.error(f"Erreur de connexion à la base de données : {e}")
        return False

# Authentification
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Page de connexion")
    st.subheader("Bienvenue dans l'outil de prédiction des ventes e-commerce")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if check_credentials(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Identifiants incorrects")
    st.stop()

# Chargement des modèles et encodeurs
model = joblib.load('random_forest_model.pkl')
label_encoder_category = joblib.load('label_encoder_category.pkl')
label_encoder_payment = joblib.load('label_encoder_payment.pkl')

# Historique des prédictions
columns = ['Catégorie', 'Mode de paiement', "Prix_d'origine_moyen", 'Remise_moyenne',
           'Année', 'Mois', 'Ventes_prévues', 'Total_annuel']
if 'historique_predictions' not in st.session_state:
    st.session_state.historique_predictions = pd.DataFrame(columns=columns)

# Interface utilisateur
st.subheader("Paramètres du produit")
col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Catégorie", label_encoder_category.classes_)
    payment_method = st.selectbox("Mode de paiement", label_encoder_payment.classes_)
    average_original_price = st.number_input("Prix original (Rs)", min_value=1.0, value=200.0)

with col2:
    average_discount = st.slider("Remise (%)", 0, 100, 10)
    year = st.selectbox("Année", list(range(2023, 2031)))
    month = st.selectbox("Mois", list(range(1, 13)))

# Préparation des données
encoded_category = label_encoder_category.transform([category])[0]
encoded_payment = label_encoder_payment.transform([payment_method])[0]
has_discount = 1 if average_discount > 0 else 0

input_data = pd.DataFrame([[average_discount, average_original_price, encoded_category,
                            encoded_payment, has_discount, year, month]],
                          columns=['Average_Discount', 'Average_Original_Price', 'Category',
                                   'Payment_Method', 'Has_Discount', 'Year', 'Month'])

# Génération de future_inputs pour la visualisation (toujours disponible)
future_inputs = pd.DataFrame([[average_discount, average_original_price, encoded_category,
                               encoded_payment, has_discount, year, m] for m in range(1, 13)],
                             columns=input_data.columns)

# Prédiction sur clic
if st.button("Prédire les ventes mensuelles"):
    prediction = model.predict(input_data)[0]
    total_annuel = model.predict(future_inputs).sum()

    nouvelle_ligne = pd.DataFrame([{
        'Catégorie': category,
        'Mode de paiement': payment_method,
        "Prix_d'origine_moyen": average_original_price,
        'Remise_moyenne': average_discount,
        'Année': year,
        'Mois': month,
        'Ventes_prévues': prediction,
        'Total_annuel': total_annuel
    }])

    st.session_state.historique_predictions = pd.concat(
        [st.session_state.historique_predictions, nouvelle_ligne], ignore_index=True
    )

# Affichage de l'historique
st.subheader("📋 Historique des prédictions")
df_hist = st.session_state.historique_predictions.copy()
if not df_hist.empty:
    st.dataframe(df_hist)
    index_to_delete = st.selectbox("Sélectionner une ligne à supprimer", df_hist.index)
    if st.button("Supprimer cette ligne"):
        st.session_state.historique_predictions.drop(index=index_to_delete, inplace=True)
        st.session_state.historique_predictions.reset_index(drop=True, inplace=True)
    if st.button("🪝 Réinitialiser l'historique"):
        st.session_state.historique_predictions = pd.DataFrame(columns=columns)

# Visualisation
st.subheader(":bar_chart: Estimation mensuelle pour l'année choisie")
pred_mensuelles = model.predict(future_inputs)
mois_labels = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']

fig1, ax1 = plt.subplots(figsize=(6, 3))
ax1.plot(mois_labels, pred_mensuelles, marker='o', color='royalblue')
ax1.set_title(f"Estimation des revenus mensuels pour {year}")
ax1.set_ylabel("Montant estimé (Rs)")
ax1.grid(True)
st.pyplot(fig1)

# Analyse de remise
st.subheader(":dart: Impact de la remise sur les ventes")
sim_discount = st.slider("Tester une autre remise (%)", 0, 100, average_discount, step=5)
sim_input = pd.DataFrame([[sim_discount, average_original_price, encoded_category,
                           encoded_payment, 1 if sim_discount > 0 else 0,
                           year, month]],
                         columns=input_data.columns)
sim_pred = model.predict(sim_input)[0]
st.write(f" Estimation avec {sim_discount}% de remise : **{sim_pred:.2f} Rs**")

discount_range = list(range(0, 101, 5))
discount_preds = model.predict(pd.DataFrame([[d, average_original_price, encoded_category,
                                              encoded_payment, 1 if d > 0 else 0,
                                              year, month] for d in discount_range],
                                            columns=input_data.columns))

fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.plot(discount_range, discount_preds, marker='o', color='green')
ax2.axvline(x=average_discount, color='red', linestyle='--', label='Remise actuelle')
ax2.set_xlabel("Remise (%)")
ax2.set_ylabel("Ventes estimées (Rs)")
ax2.set_title("Effet de la remise sur le revenu mensuel")
ax2.legend()
st.pyplot(fig2)

# Export PDF
df_hist = st.session_state.historique_predictions.copy()
if fpdf_installed and not df_hist.empty:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)
    pdf.cell(190, 8, txt="Historique des prédictions de ventes", ln=True, align='C')
    pdf.ln(4)

    column_labels = {
        'Catégorie': 'Catégorie',
        'Mode de paiement': 'Paiement',
        "Prix_d'origine_moyen": 'Prix moyen',
        'Remise_moyenne': 'Remise (%)',
        'Année': 'Année',
        'Mois': 'Mois',
        'Ventes_prévues': 'Ventes prévues',
        'Total_annuel': 'Total annuel'
    }
    for col in columns:
        pdf.cell(24, 6, column_labels[col], border=1)
    pdf.ln()
    for _, row in df_hist.iterrows():
        for col in columns:
            val = f"{row[col]:.4f}" if isinstance(row[col], (float, np.floating)) else str(row[col])
            pdf.cell(24, 6, val, border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.cell(190, 8, txt=f"Total cumulé des ventes prévues : {df_hist['Ventes_prévues'].sum():.2f} Rs", ln=True)
    pdf.cell(190, 8, txt=f"Total cumulé annuel : {df_hist['Total_annuel'].sum():.2f} Rs", ln=True)

    pdf_path = "resume_prediction_global.pdf"
    pdf.output(pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("🗓️ Télécharger les résultats (PDF)", f, file_name="resume_prediction_global.pdf")
elif not fpdf_installed:
    st.warning("Le module `fpdf` n'est pas installé. Utilisez `pip install fpdf` pour activer l'export PDF.")
