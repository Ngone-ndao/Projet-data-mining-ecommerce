import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from datetime import timedelta
import ast
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, silhouette_samples
import re
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def safe_dataframe(df):
    df_display = df.copy()
    df_display.columns = df_display.columns.astype(str)
    st.dataframe(df_display, width="stretch")


def clean_dataframe_for_streamlit(df):
    """
    Nettoie un dataframe pour l'affichage Streamlit
    """
    if df is None or df.empty:
        return df
    
    df_clean = df.copy()
    
    # Convertir toutes les colonnes problématiques
    for col in df_clean.columns:
        # Si c'est une colonne object, convertir en string
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
    
    # Colonnes spécifiques
    for col in ['StockCode', 'Type', 'Description', 'Country']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
    
    # Remplacer NaN
    df_clean = df_clean.fillna('')
    
    return df_clean

# ===============================
# CONFIGURATION INITIALE ET STYLES
# ===============================
st.set_page_config(
    page_title="Dashboard Data Mining",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    /* Style général */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    /* Cartes de métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 30px 0;
    }
    
    /* Cards */
    .custom-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# INITIALISATION
# ===============================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "À propos de nous"
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'kmeans_data' not in st.session_state:
    st.session_state.kmeans_data = None

# ===============================
# FONCTION POUR LA PAGE D'ACCUEIL
# ===============================
def home_page():
    # Header avec bannière
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin-bottom: 40px;'>
            <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700;'>📊 Dashboard Data Mining</h1>
            <p style='font-size: 1.3rem; opacity: 0.95; margin-top: 10px;'>Analyse avancée des données clients e-commerce</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("## 🎯 Objectifs du Projet")
    st.markdown("""
    Cette application permet d'analyser les données transactionnelles pour fournir des insights actionnables
    et optimiser les stratégies marketing.
    """)
    
    # Cartes d'objectifs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea;'>🔍 Découverte de Patterns</h3>
            <p>Identification des associations entre produits avec l'algorithme FP-Growth pour des recommandations intelligentes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea;'>👥 Segmentation Clients</h3>
            <p>Clustering avec K-means et segmentation RFM pour comprendre les comportements d'achat.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea;'>📈 Décision Marketing</h3>
            <p>Recommandations personnalisées et stratégies ciblées pour chaque segment de clients.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Guide rapide
    with st.expander("🚀 **Comment utiliser l'application**", expanded=True):
        st.markdown("""
        ### 📋 Guide d'utilisation
        
        1. **📁 Importez vos données** via le panneau latéral (format Excel requis)
        2. **📊 Explorez les données** avec les statistiques descriptives
        3. **📈 Visualisez** les tendances et patterns
        4. **🤖 Choisissez un modèle** pour l'analyse :
           - **🛒 FP-Growth** : Recommandations produits basées sur les associations
           - **👥 K-means** : Segmentation comportementale des clients
           - **⭐ RFM** : Analyse de fidélité et valeur client
        5. **📑 Consultez le résumé** pour une vue d'ensemble des insights
        """)
    
    # Technologies utilisées
    st.markdown("## 🛠️ Technologies Utilisées")
    tech_cols = st.columns(5)
    technologies = [
        ("Python", "🐍", "#3776AB"),
        ("Streamlit", "🎈", "#FF4B4B"),
        ("Pandas", "🐼", "#150458"),
        ("Scikit-learn", "🧠", "#F7931E"),
        ("Plotly", "📈", "#3F4F75")
    ]
    
    for idx, (tech, emoji, color) in enumerate(technologies):
        with tech_cols[idx]:
            st.markdown(f"""
            <div style='background-color: {color}; color: white; padding: 20px; 
                        border-radius: 10px; text-align: center; height: 100px; display: flex; 
                        flex-direction: column; justify-content: center;'>
                <h2 style='margin: 0;'>{emoji}</h2>
                <p style='margin: 10px 0 0 0; font-weight: bold; font-size: 1.1rem;'>{tech}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistiques rapides si des données sont chargées
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("## 📊 Aperçu des Données Chargées")
        
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Nombre de Transactions",
                value=f"{len(df):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Clients Uniques",
                value=f"{df['CustomerID'].nunique():,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Produits Uniques",
                value=f"{df['Description'].nunique():,}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Chiffre d'Affaires",
                value=f"€{df['Montant'].sum():,.0f}",
                delta=None
            )

# ===============================
# SIDEBAR AMÉLIORÉ
# ===============================
def create_sidebar():
    with st.sidebar:
        # Logo et titre
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #1E3A8A; font-size: 1.8rem; margin-bottom: 5px;'>📊 Data Mining</h1>
            <p style='color: #666; font-size: 1rem;'>Dashboard E-commerce</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload de fichier stylisé
        st.markdown("### 📁 Import des Données")
        uploaded_file = st.file_uploader(
            "Choisissez un fichier Excel",
            type=["xlsx"],
            help="Format attendu : fichier Excel avec données transactionnelles",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Check if the uploaded file is different from the one currently in session state
            if st.session_state.df is None or uploaded_file.name != st.session_state.last_uploaded_filename:
                try:
                    # Traitement selon le type de fichier
                    if uploaded_file.name.endswith('.xlsx'):
                        df_temp = pd.read_excel(uploaded_file)

                        # Convertir StockCode en string si la colonne existe
                        if 'StockCode' in df_temp.columns:
                           df_temp['StockCode'] = df_temp['StockCode'].astype(str)
            
                        # Convertir Type en string si la colonne existe
                        if 'Type' in df_temp.columns:
                            df_temp['Type'] = df_temp['Type'].astype(str)

                          # 3. Convertir TOUTES les colonnes 'object' en string
                        for col in df_temp.select_dtypes(include=['object']).columns:
                            df_temp[col] = df_temp[col].astype(str)

                          # 4. Remplacer NaN par None
                        df_temp = df_temp.replace({np.nan: None})

                        # Calculate 'Montant' and store in session state
                        df_temp['Montant'] = df_temp['Quantity'] * df_temp['UnitPrice']
                        st.session_state.df = df_temp
                        st.session_state.last_uploaded_filename = uploaded_file.name
                        st.success("✅ Fichier importé avec succès")
                    else:
                        st.error("❌ Format de fichier non supporté.")
                        st.session_state.df = None
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'importation : {e}")
                    st.session_state.df = None
        
        st.markdown("---")
        
        # Navigation avec icônes
        st.markdown("### 🧭 Navigation")
        
        menu_options = {
            "🏠 Accueil": "home",
            "📋 Description": "description",
            "📊 Visualisation": "visualisation",
            "🤖 Modélisation": "modelisation",
            "📈 Résumé": "resume",
            "👥 À propos de nous": "about"
        }
        
        # Créer des boutons pour la navigation
        for label, key in menu_options.items():
            if st.button(label, key=key, width="stretch"):
                st.session_state.menu_choice = label
        
        st.markdown("---")
        
        # Information sur les données
        if st.session_state.df is not None:
            st.markdown("### ℹ️ Informations Données")
            df = st.session_state.df
            
            # Afficher les métriques dans la sidebar
            st.caption(f"📊 **Lignes :** {df.shape[0]:,}")
            st.caption(f"📈 **Colonnes :** {df.shape[1]}")
            st.caption(f"👥 **Clients :** {df['CustomerID'].nunique():,}")
            st.caption(f"💰 **CA Total :** €{df['Montant'].sum():,.0f}")
        
        # Footer de la sidebar
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 10px;'>
            Projet Data Mining 2026<br>
            Master Data Science
        </div>
        """, unsafe_allow_html=True)

# ===============================
# DESCRIPTION DES DONNÉES 
# ===============================
def description_data():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 30px;'>
        <h2 style='margin: 0;'>📋 Analyse Descriptive des Données</h2>
        <p style='opacity: 0.9;'>Explorez et comprenez la structure de vos données</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df
    if df is not None:
        # Onglets pour organiser le contenu
        tab1, tab2, tab3, tab4 = st.tabs(["👀 Aperçu", "📊 Statistiques", "🔍 Valeurs Manquantes", "💾 Export"])
        
        with tab1:
            st.subheader("Aperçu des Données")
            col1, col2 = st.columns([3, 1])
            with col1:
                safe_dataframe(df.head(20))
            with col2:
                st.metric("Lignes", f"{df.shape[0]:,}")
                st.metric("Colonnes", df.shape[1])
                st.metric("Taille Mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Informations sur les colonnes
            with st.expander("📋 Informations sur les colonnes"):
                col_info = pd.DataFrame({
                    'Colonne': df.columns,
                    'Type': df.dtypes.astype(str).values,
                    'Valeurs Uniques': [df[col].nunique() for col in df.columns],
                    'Valeurs Null': df.isnull().sum().values
                })
                safe_dataframe(col_info)
        
        with tab2:
            st.subheader("Statistiques Descriptives")
            
            # Sélection des colonnes numériques
            num_cols = [col for col in ['Quantity', 'UnitPrice', 'Montant'] if col in df.columns]

            
            if num_cols:
                 # Statistiques détaillées
                stats_df = df[num_cols].describe().T
                stats_df['skew'] = df[num_cols].skew()
                stats_df['kurtosis'] = df[num_cols].kurtosis()

                  # Arrondir pour l'affichage
                stats_df = stats_df.round(2)

                  # Affichage sécurisé
                safe_dataframe(stats_df)
                
                # Distribution des variables numériques
                st.subheader("📈 Distributions")
                selected_col = st.selectbox("Sélectionnez une variable", num_cols)
                if selected_col:
                    df_plot = df[[selected_col]].dropna()
                    fig = px.histogram(df_plot, x=selected_col, nbins=50, 
                                      title=f"Distribution de {selected_col}",
                                      color_discrete_sequence=['#667eea'])
                    st.plotly_chart(fig, width="stretch")
        
        with tab3:
            st.subheader("Analyse des Valeurs Manquantes")
            
            # Métriques
            col1, col2, col3 = st.columns(3)
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            with col1:
                st.metric("Cellules Total", f"{total_cells:,}")
            with col2:
                st.metric("Valeurs Manquantes", f"{missing_cells:,}")
            with col3:
                st.metric("% Manquant", f"{missing_percentage:.2f}%")
            
            # Visualisation des valeurs manquantes
            missing_df = pd.DataFrame({
                'Colonne': df.columns,
                'Valeurs Manquantes': df.isnull().sum().values,
                '% Manquant': (df.isnull().sum().values / len(df)) * 100
            }).sort_values('% Manquant', ascending=False)
            
            fig = px.bar(missing_df.head(10), x='Colonne', y='% Manquant',
                        title='Top 10 des colonnes avec valeurs manquantes',
                        color='% Manquant',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, width="stretch")
            
            # Bouton pour nettoyer les données
            if st.button("🧹 Nettoyer les données (supprimer les valeurs manquantes)", type="primary"):
              df_before = st.session_state.df
              df_cleaned = df_before.dropna().copy()
    
              st.session_state.df = df_cleaned
    
              st.success(
              f"✅ Données nettoyées ! "
              f"{len(df_before) - len(df_cleaned)} lignes supprimées."
             )
              st.rerun()
        
        with tab4:
            st.subheader("Export des Données")
            st.markdown("Téléchargez les données nettoyées pour une utilisation externe.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger en CSV",
                    data=csv,
                    file_name="donnees_nettoyees.csv",
                    mime="text/csv",
                    width="stretch"
                )
            
            with col2:
                # Export Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Donnees')
                st.download_button(
                    label="📥 Télécharger en Excel",
                    data=output.getvalue(),
                    file_name="donnees_nettoyees.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )
    else:
        st.warning("Veuillez d'abord importer un fichier de données.")

# ===============================
# VISUALISATION AMÉLIORÉE 
# ===============================
def visualize_data():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 30px;'>
        <h2 style='margin: 0;'>📊 Visualisation des Données</h2>
        <p style='opacity: 0.9;'>Explorez les tendances et patterns dans vos données</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df
    if df is not None:
        # Onglets pour différentes visualisations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performances", "👥 Clients", "🛒 Produits", "⏰ Temporalité"])
        
        with tab1:
            st.subheader("📈 Analyse des Performances")
            
            # Métriques clés
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ca_total = df['Montant'].sum()
                st.metric("Chiffre d'Affaires Total", f"€{ca_total:,.0f}")
            with col2:
                panier_moyen = df.groupby('InvoiceNo')['Montant'].sum().mean()
                st.metric("Panier Moyen", f"€{panier_moyen:,.2f}")
            with col3:
                transactions = df['InvoiceNo'].nunique()
                st.metric("Transactions", f"{transactions:,}")
            with col4:
                clients = df['CustomerID'].nunique()
                st.metric("Clients Actifs", f"{clients:,}")
            
            # Évolution du CA dans le temps
            st.subheader("📅 Évolution du Chiffre d'Affaires")
            
            # Préparation des données temporelles
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Date'] = df['InvoiceDate'].dt.date
            df['Mois'] = df['InvoiceDate'].dt.to_period('M').astype(str)
            
            # Sélection de la période
            period = st.radio("Période d'analyse", ["Journalier", "Hebdomadaire", "Mensuel"], horizontal=True)
            
            if period == "Journalier":
                time_data = df.groupby('Date')['Montant'].sum().reset_index()
                title = "Évolution Journalière du Chiffre d'Affaires"
                x_col = 'Date'
            elif period == "Hebdomadaire":
                df['Semaine'] = df['InvoiceDate'].dt.isocalendar().year.astype(str) + '-W' + df['InvoiceDate'].dt.isocalendar().week.astype(str).str.zfill(2)
                time_data = df.groupby('Semaine')['Montant'].sum().reset_index()
                title = "Évolution Hebdomadaire du Chiffre d'Affaires"
                x_col = 'Semaine'
            else:  # Mensuel
                time_data = df.groupby('Mois')['Montant'].sum().reset_index()
                title = "Évolution Mensuelle du Chiffre d'Affaires"
                x_col = 'Mois'
            
            # Graphique d'évolution
            fig = px.line(time_data, x=x_col, y='Montant',
                         title=title,
                         markers=True,
                         line_shape='spline',
                         color_discrete_sequence=['#667eea'])
            
            fig.update_layout(
                xaxis_title="Période",
                yaxis_title="Chiffre d'Affaires (€)",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, width="stretch")
        
        with tab2:
            st.subheader("👥 Analyse des Clients")
            
            # Top clients
            top_clients = df.groupby('CustomerID').agg({
                'Montant': 'sum',
                'InvoiceNo': 'nunique',
                'Quantity': 'sum'
            }).rename(columns={
                'Montant': 'CA_Total',
                'InvoiceNo': 'Nb_Commandes',
                'Quantity': 'Quantité_Totale'
            }).nlargest(5, 'CA_Total')
            
            # Graphique des top clients
            fig1 = px.bar(top_clients.reset_index(), 
                         x='CustomerID', y='CA_Total',
                         title='🏆 Top 5 Clients par Chiffre d\'Affaires',
                         color='CA_Total',
                         color_continuous_scale='Viridis',
                         labels={'CA_Total': 'CA Total (€)', 'CustomerID': 'ID Client'})
            
            st.plotly_chart(fig1, width="stretch")

            # Comptage du nombre d'achats par client (nombre de factures uniques)
            clients_top_achats = (
              df.groupby("CustomerID")["InvoiceNo"]
             .nunique()
             .reset_index()
             .sort_values(by="InvoiceNo", ascending=False)
             .head(5)
             )

           # Création du graphique Plotly
            fig2 = px.bar(
            clients_top_achats,
            x="CustomerID",
            y="InvoiceNo",
            text="InvoiceNo",
            title="👥 Top 5 des clients ayant fait plus d’achats",
            color="CustomerID",
            color_discrete_sequence=px.colors.qualitative.Set2
            )

             # Ajout des étiquettes de valeur
            fig2.update_traces(texttemplate='%{text}', textposition='outside')

             # Mise en forme du graphique
            fig2.update_layout(
             xaxis_title="ID Client",
             yaxis_title="Nombre d'achats",
             uniformtext_minsize=8,
             uniformtext_mode='hide'
          )

             # Affichage dans Streamlit
            st.plotly_chart(fig2)
            
            # Distribution du CA par client
            ca_par_client = df.groupby('CustomerID')['Montant'].sum()
            
            col1, col2 = st.columns(2)
            with col1:
                fig3 = px.histogram(ca_par_client, nbins=50,
                                   title='Distribution du CA par Client',
                                   labels={'value': 'CA Client (€)', 'count': 'Nombre de Clients'},
                                   color_discrete_sequence=['#764ba2'])
                st.plotly_chart(fig3, width="stretch")
            
            with col2:
                # Box plot du CA par client
                fig4 = px.box(ca_par_client.reset_index(), y='Montant',
                             title='Distribution du CA (Box Plot)',
                             labels={'Montant': 'CA Client (€)'},
                             color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig4, width="stretch")
        
        with tab3:
            st.subheader("🛒 Analyse des Produits")
            
            # Top produits
            top_products = df.groupby('Description').agg({
                'Quantity': 'sum',
                'Montant': 'sum',
                'InvoiceNo': 'nunique'
            }).rename(columns={
                'Quantity': 'Quantité_Vendue',
                'Montant': 'CA_Produit',
                'InvoiceNo': 'Nb_Transactions'
            }).nlargest(5, 'CA_Produit')
            
            # Graphique des top produits
            fig1 = px.bar(top_products.reset_index(), 
                         x='Description', y='CA_Produit',
                         title='🏆 Top 5 Produits par Chiffre d\'Affaires',
                         color='CA_Produit',
                         color_continuous_scale='Plasma',
                         labels={'CA_Produit': 'CA Produit (€)', 'Description': 'Produit'})
            
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, width="stretch")

            # Préparation des données pour les graphiques
            top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(5).reset_index()

            # Création du graphique Plotly
            fig2 = px.bar(
              top_products,
              x="Description",
              y="Quantity",
              orientation='v',
              title="🎯 Top 5 produits vendus",
              color='Description',
              color_discrete_sequence=px.colors.qualitative.Set3
           )

            # Affichage dans Streamlit
            st.plotly_chart(fig2)
            
            # Relation quantité vs prix
            st.subheader("📦 Relation Quantité vs Prix")
            
            product_stats = df.groupby('Description').agg({
                'Quantity': 'sum',
                'UnitPrice': 'mean',
                'Montant': 'sum'
            }).reset_index()
            
            # Filtrer les valeurs négatives
            product_stats_filtered = product_stats[
                (product_stats['Montant'] > 0) & 
                (product_stats['Quantity'] > 0) & 
                (product_stats['UnitPrice'] > 0)
            ]
            
            if not product_stats_filtered.empty:
                # Créer une colonne pour la taille (valeur absolue avec minimum)
                product_stats_filtered['Size'] = product_stats_filtered['Montant'].abs()
                product_stats_filtered['Size'] = product_stats_filtered['Size'].clip(lower=1)
                
                fig3 = px.scatter(product_stats_filtered, x='UnitPrice', y='Quantity',
                                 size='Size',
                                 color='Montant',
                                 hover_name='Description',
                                 title='Relation Prix Moyen vs Quantité Vendue (ventes positives uniquement)',
                                 labels={'UnitPrice': 'Prix Moyen (€)', 'Quantity': 'Quantité Vendue'},
                                 color_continuous_scale='Viridis',
                                 log_x=True, log_y=True)
                
                st.plotly_chart(fig3, width="stretch")
                st.caption("ℹ️ Note : Les retours et annulations (montants négatifs) ont été exclus de cette visualisation.")
            else:
                st.warning("Aucune donnée positive disponible pour cette visualisation.")

                st.subheader("📊 Quantité moyenne vendue par classe de prix")

            # Création des classes de prix (quartiles)
            product_stats_filtered['PriceClass'] = pd.qcut(
            product_stats_filtered['UnitPrice'],
            q=4,
            labels=['Bas', 'Moyen-bas', 'Moyen-haut', 'Élevé']
           )

            # Agrégation
            price_quantity = product_stats_filtered.groupby('PriceClass').agg({
               'Quantity': 'mean',
               'Montant': 'sum'
           }).reset_index()

            # Graphique
            fig_bar = px.bar(
              price_quantity,
              x='PriceClass',
              y='Quantity',
              text=price_quantity['Quantity'].round(0),
              color='PriceClass',
              title="Quantité moyenne vendue selon la classe de prix",
              labels={
             'PriceClass': 'Classe de prix',
             'Quantity': 'Quantité moyenne vendue'
            }
          )       

            fig_bar.update_layout(showlegend=False)
            fig_bar.update_traces(textposition='outside')

            st.plotly_chart(fig_bar, width="stretch")

            st.caption(
              "ℹ️ Les produits sont regroupés par classes de prix (quartiles). "
              "Ce graphique permet de confirmer la relation entre prix et volume des ventes."
          )

        
        with tab4:
            st.subheader("⏰ Analyse Temporelle")
            
            # Analyse par jour de la semaine
            df['Jour_Semaine'] = df['InvoiceDate'].dt.day_name()
            df['Heure'] = df['InvoiceDate'].dt.hour
            
            # Ordre des jours
            jours_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            jours_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            
            ca_par_jour = df.groupby('Jour_Semaine')['Montant'].sum().reindex(jours_order)
            ca_par_jour.index = jours_fr
            
            # Graphique par jour
            fig1 = px.bar(x=ca_par_jour.index, y=ca_par_jour.values,
                         title='📅 Chiffre d\'Affaires par Jour de la Semaine',
                         labels={'x': 'Jour', 'y': 'Chiffre d\'Affaires (€)'},
                         color=ca_par_jour.values,
                         color_continuous_scale='Blues')
            
            st.plotly_chart(fig1, width="stretch")
            
            # Carte thermique jour/heure
            st.subheader("🔥 Carte Thermique des Ventes")
            
            heatmap_data = df.pivot_table(
                values='Montant',
                index='Jour_Semaine',
                columns='Heure',
                aggfunc='sum'
            ).reindex(jours_order)
            
            heatmap_data.index = jours_fr
            
            fig2 = px.imshow(heatmap_data,
                            title='Carte Thermique des Ventes par Jour et Heure',
                            labels=dict(x="Heure", y="Jour", color="CA (€)"),
                            aspect='auto',
                            color_continuous_scale='YlOrRd')
            
            st.plotly_chart(fig2, width="stretch")
            
            # Analyse par heure
            ca_par_heure = df.groupby('Heure')['Montant'].sum()
            
            fig3 = px.line(x=ca_par_heure.index, y=ca_par_heure.values,
                          title='⏰ Chiffre d\'Affaires par Heure de la Journée',
                          labels={'x': 'Heure', 'y': 'Chiffre d\'Affaires (€)'},
                          markers=True)
            
            fig3.update_layout(xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig3, width="stretch")


             # Extraction du jour de la semaine
             # Ensure 'InvoiceDate' is datetime before operations
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['Num_Jour'] = df['InvoiceDate'].dt.dayofweek

             # Groupement et comptage
            order_day = df.groupby('Num_Jour')['InvoiceNo'].nunique()

             # Création du graphique avec matplotlib/seaborn
            st.subheader("📊 Nombre de Transactions par Jour de la Semaine")
             # Initialisation de la figure
            fig4, ax = plt.subplots(figsize=(12, 8))

            # Tracé du graphique
            sns.barplot(x=order_day.index, y=order_day.values, palette="Set3", ax=ax)
            ax.set_title('Nombre de Transactions par Jour', size=20)
            ax.set_xlabel('Jour', size=14)
            ax.set_ylabel('Nombre de Transactions', size=14)
            ax.xaxis.set_tick_params(labelsize=11)
            ax.yaxis.set_tick_params(labelsize=11)
            ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])

             # Affichage dans Streamlit
            st.pyplot(fig4)


             # S'assurer que 'InvoiceDate' est bien au format datetime
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

             # Extraire uniquement la date (sans l'heure)
            df["Date"] = df["InvoiceDate"].dt.date

    else:
        st.warning("Veuillez d'abord importer un fichier de données.")

        

# ===============================
# MODÉLISATION 
# ===============================
def modeling_and_predictions():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 30px;'>
        <h2 style='margin: 0;'>🤖 Modélisation et Prédictions</h2>
        <p style='opacity: 0.9;'>Appliquez des algorithmes de data mining pour extraire des insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.df
    if df is None:
        st.warning("Veuillez d'abord importer un fichier de données pour la modélisation.")
        return
    
      # Menu de sélection du modèle
    menu1 = ["👥 K-means", "⭐ Segmentation RFM", "🛒 FP_GROWTH"] 
    choix = st.sidebar.selectbox('Choisissez une méthode', menu1, key='modeling_menu_selection')
    
    # Préparation des données pour les modèles
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df_invoice = df.groupby(['InvoiceNo', 'InvoiceDate', 'CustomerID']).agg({
           'Quantity': 'sum',
           'Montant': 'sum'
        }).reset_index()

         # Date d'analyse
    analysis_date = df_invoice['InvoiceDate'].max() + timedelta(days=1)
    
        # Section d'information
    with st.expander("📊 Informations sur les données de modélisation", expanded=False):
          st.write(f"**Date d'analyse :** {analysis_date.date()}")
          st.write(f"**Période couverte :** {df['InvoiceDate'].min().date()} au {df['InvoiceDate'].max().date()}")
          st.write(f"**Nombre de clients uniques :** {df['CustomerID'].nunique():,}")
          st.write(f"**Nombre de transactions :** {df['InvoiceNo'].nunique():,}")
    
    # K-MEANS 
    def kmeans_clustering(df_invoice, analysis_date):
        st.markdown("""
          <div style='background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #1E3A8A; margin: 0;'>👥 Segmentation Client avec K-means</h3>
          <p style='margin: 5px 0 0 0;'>Regroupement des clients en clusters basé sur leur comportement d'achat.</p>
         </div>
         """, unsafe_allow_html=True)
     
        with st.expander("ℹ️ Explication du modèle", expanded=False):
          st.markdown("""
            **K-means** est un algorithme de clustering non supervisé qui partitionne les données en K clusters.
        
            **Méthodes de sélection du nombre optimal de clusters :**
        
           - **📉 Méthode du coude** : Analyse la diminution de l'inertie
           - **📊 Score de silhouette** : Mesure la cohésion et séparation des clusters
           - **⚖️ Score ARI** : Évalue la stabilité des clusters
           """)
    
          # Préparation des données 
          base = df_invoice.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'Montant': 'sum'
          }).rename(columns={
           'InvoiceDate': 'Recence',
           'InvoiceNo': 'Frequence',
           'Quantity': 'Quantite_totale',
           'Montant': 'Montant'
        }).reset_index()
     
    
        # Afficher quelques statistiques pour vérifier
        with st.expander("📊 Vérification des données RFM"):
          st.write(f"Nombre de clients: {len(base)}")
          st.write(f"Moyenne Récence: {base['Recence'].mean():.1f} jours")
          st.write(f"Moyenne Fréquence: {base['Frequence'].mean():.1f}")
          st.write(f"Moyenne Montant: €{base['Montant'].mean():.2f}")
          safe_dataframe(
           base[['Recence', 'Frequence', 'Quantite_totale', 'Montant']].describe()
       )

    
         # Normalisation
        scaler = StandardScaler()
        base_scaled = scaler.fit_transform(base[['Recence', 'Frequence', 'Quantite_totale', 'Montant']])
    
         # Stocker dans session state pour accès global
        st.session_state.kmeans_data = {
          'base': base,
          'base_scaled': base_scaled,
          'scaler': scaler
        }
          # Vérifier la normalisation
        st.success(f"✅ Données préparées: {len(base)} clients, 4 variables normalisées")
    
        KMEANS_PARAMS = {
          'init': 'k-means++',
          'max_iter': 300,
          'n_init': 10,
          'random_state': 42,
          'algorithm':'lloyd'
        
           }
     
        st.sidebar.info(f"🔧 Paramètres KMeans: {KMEANS_PARAMS}")

      # Onglets pour l'analyse K-means
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Méthode du Coude", "📊 Score Silhouette", "👥 Clustering", "📋 Résultats"])
    
        with tab1:
           st.subheader("Méthode du Coude pour Déterminer k Optimal")

           inertia = []
           k_max = 15

           progress_bar = st.progress(0)
           status_text = st.empty()

           with st.spinner("Calcul de l'inertie pour différents k..."):
             for k in range(1, k_max + 1):
                status_text.text(f"Calcul pour k = {k}...")
                kmeans = KMeans(n_clusters=k, **KMEANS_PARAMS)
                
            
                kmeans.fit(base_scaled)
                inertia.append(kmeans.inertia_)
                progress_bar.progress(k / k_max)

                status_text.text("✅ Calcul terminé")

                # =======================
                # 📊 GRAPHIQUE DU COUDE
                # =======================
             fig, ax = plt.subplots(figsize=(10, 7))
             ax.plot(range(1, k_max + 1), inertia, marker='o', linewidth=2)
             ax.set_xticks(range(1, k_max + 1))
             ax.set_xlabel("Nombre de clusters (k)", fontsize=13)
             ax.set_ylabel("Inertie", fontsize=13)
             ax.set_title("Méthode du coude", fontsize=15, fontweight="bold")
             ax.grid(alpha=0.3)

             # Zone recommandée
             ax.axvspan(5, 6, color="orange", alpha=0.2, label="Zone optimale (5–6)")
             ax.legend()

             st.pyplot(fig)

             # =======================
             # 🧮 ANALYSE NUMÉRIQUE
             # =======================
             differences = np.diff(inertia)
             ratios = differences[1:] / differences[:-1]

             elbow_auto = np.argmin(ratios) + 2  # coude mathématique
             elbow_k = elbow_auto

              # =======================
              # 📝 INTERPRÉTATION
              # =======================
             st.subheader("📝 Interprétation de la méthode du coude")

             st.markdown(f"""
             *Analyse automatique :*
             - Le premier ralentissement mathématique est détecté à *k = {elbow_auto}*
             - Cette valeur correspond à une structure globale des données

             *Analyse visuelle et métier :*
             - La diminution de l'inertie reste significative jusqu'à *k = 5–6*
             - À partir de *k ≥ 6*, le gain marginal devient faible
             - Des valeurs de k inférieures (k ≤ 3) produisent une segmentation trop grossière

              ### ✅ Conclusion
             > La méthode du coude suggère *une zone optimale comprise entre 5 et 6 clusters*,  
             > offrant un bon compromis entre performance statistique et interprétabilité métier.
             """)

              # =======================
              # 📋 TABLEAU RÉCAPITULATIF
              # =======================
             st.subheader("📊 Valeurs d'inertie")
             inertia_df = pd.DataFrame({
                 "k": range(1, k_max + 1),
                 "Inertie": inertia,
                 "Δ Inertie": [np.nan] + list(differences),
                 "% Δ": [np.nan] + list((differences / inertia[:-1]) * 100)
               })

             def highlight_zone(row):
                if row["k"] in [5, 6]:
                 return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

             safe_dataframe(inertia_df.round(2))
     

        with tab2:
          st.subheader("Score de silhouette pour validation")
        
          st.info("📊 Calcul des scores de silhouette en cours...")
        
           # Barre de progression
          progress_bar_sil = st.progress(0)
          status_text_sil = st.empty()
          silhouette_scores = []
          best_k = 5  # Valeur par défaut
          best_score = -1
        
        
          for i in range(3, k_max + 1):
            status_text_sil.text(f"Calcul du score silhouette pour k = {i}...")
            
            # UTILISER LES MÊMES PARAMÈTRES EXACTS
            kmeans = KMeans(n_clusters=i, **KMEANS_PARAMS)
            labels = kmeans.fit_predict(base_scaled)
            score = silhouette_score(base_scaled, labels)
            silhouette_scores.append(score)
            
            # Garder le meilleur score
            if score > best_score:
                best_score = score
                best_k = i
            
            progress_bar_sil.progress((i-1) / (k_max-1))
        
          status_text_sil.text("✓ Calcul des scores de silhouette terminé!")
        
          # Trouver le meilleur score
          best_k_silhouette = best_k
          best_score_silhouette = best_score
        
          # Debug: Afficher tous les scores
          with st.expander("🔍 Voir tous les scores de silhouette", expanded=False):
             for k, score in zip(range(3, k_max + 1), silhouette_scores):
                st.write(f"k = {k}: {score:.4f}")
        
          # Visualisation
          fig2, ax2 = plt.subplots()
          fig2.set_size_inches(10, 7)
          plt.rcParams['font.size'] = 14
        
          ax2.plot(range(3, k_max + 1), silhouette_scores, 'o-', color='#E74C3C', linewidth=2, markersize=8)
          ax2.set_xticks(np.arange(3, k_max + 1, 1))
          ax2.set_title('Score de silhouette', fontsize=16, fontweight='bold', pad=20)
          ax2.set_xlabel('Nombre de classes (k)', fontsize=14)
          ax2.set_ylabel('Score de silhouette', fontsize=14)
          ax2.grid(True, alpha=0.3)
        
          # Marquer le meilleur score
          ax2.axvline(x=best_k_silhouette, color='#27AE60', linestyle='--', alpha=0.7, linewidth=2)
          ax2.plot(best_k_silhouette, best_score_silhouette, 'o', markersize=12, 
                markeredgecolor='black', markerfacecolor='#27AE60')
        
            # Ajouter le texte du meilleur score
          ax2.text(best_k_silhouette, best_score_silhouette + 0.01, 
                f'Optimal: k={best_k_silhouette}\nScore={best_score_silhouette:.3f}', 
                color='#27AE60', ha='center', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
             # Lignes de référence pour l'interprétation
          ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
          ax2.axhline(y=0.7, color='green', linestyle=':', alpha=0.5)
        
           # Ajouter les valeurs sur le graphique
          for i, (x, y) in enumerate(zip(range(3, k_max + 1), silhouette_scores)):
            if i in [0, 3, 4, 5, 8, 13]:  # Afficher des valeurs clés
                ax2.text(x, y, f'{y:.3f}', fontsize=9, ha='center', va='bottom')
        
          st.pyplot(fig2)
        
          # Analyse détaillée
          st.subheader("📊 Analyse des scores")
        
          # Créer un tableau des scores
          scores_data = []
          for k, score in zip(range(3, k_max + 1), silhouette_scores):
            interpretation = ""
            if score >= 0.7:
                interpretation = "Structure forte 🟢"
            elif score >= 0.5:
                interpretation = "Structure raisonnable 🟡"
            elif score >= 0.25:
                interpretation = "Structure faible 🟠"
            else:
                interpretation = "Pas de structure 🔴"
            
            scores_data.append({
                'k': k,
                'Score': score,
                'Interprétation': interpretation,
                'Différence avec le meilleur': best_score_silhouette - score
            })
        
          scores_df = pd.DataFrame(scores_data)
        
           # Mettre en évidence le meilleur
          def highlight_best(row):
            if row['k'] == best_k_silhouette:
                return ['background-color: #d4edda' for _ in range(len(row))]
            elif row['Score'] >= 0.5:
                return ['background-color: #fff3cd' for _ in range(len(row))]
            return [''] * len(row)
        
          st.dataframe(
            scores_df.style.format({
                'Score': '{:.4f}',
                'Différence avec le meilleur': '{:.4f}'
            }).apply(highlight_best, axis=1),
            width="stretch"
          )
        
           # Interprétation
          st.subheader("🎯 Interprétation")
        
          interpretation_text = ""
          if best_score_silhouette >= 0.7:
              interpretation_text = "**Excellente** structure de clusters"
          elif best_score_silhouette >= 0.5:
              interpretation_text = "**Bonne** structure de clusters"
          elif best_score_silhouette >= 0.25:
              interpretation_text = "Structure **faible** mais acceptable"
          else:
              interpretation_text = "Structure **inappropriée**"
        
          st.markdown(f"""
          **Résultats :**
          - **Meilleur k** : {best_k_silhouette}
          - **Score optimal** : {best_score_silhouette:.4f}
          - **Interprétation** : {interpretation_text}
        
          **Signification :**
          Pour k = {best_k_silhouette}, le score de silhouette est maximal, ce qui signifie :
          1. **Bonne cohésion** : Les points dans chaque cluster sont proches
          2. **Bonne séparation** : Les clusters sont bien distincts
          3. **Structure optimale** : Le partitionnement est le plus cohérent
        
          **Validation :**
          > Le score de silhouette confirme que **k = {best_k_silhouette}** est le choix optimal
          """)
        
          # Synthèse avec la méthode du coude
          st.subheader("🔍 Synthèse des deux méthodes")
        
          # Utiliser le coude calculé précédemment
          if 'elbow_k' in locals():
             coude_k = elbow_k
          else:
             # Calculer un coude approximatif si non calculé
             coude_k = 5  # Valeur par défaut basée sur votre analyse
        
             # Déterminer la recommandation finale
          if abs(coude_k - best_k_silhouette) <= 1:
              # Les méthodes concordent (différence de 1 ou moins)
             final_k = best_k_silhouette
            
             st.success(f"""
             ✅ **Convergence parfaite des méthodes !**
            
              **1. Méthode du coude :**
              - Suggère un intervalle optimal : **k = {coude_k}**
              - Point où l'inertie diminue moins significativement
            
              **2. Score de silhouette :**
              - Optimal à **k = {best_k_silhouette}**
              - Score : **{best_score_silhouette:.4f}** (structure {'forte' if best_score_silhouette >= 0.7 else 'raisonnable'})
            
              **🎯 Décision finale :**
              > En combinant les deux approches, le nombre de clusters retenu est **k = {final_k}**
              """)
          else:
            st.warning(f"""
            ⚠️ **Attention : Divergence entre les méthodes**
            
            - Coude suggère : k = {coude_k}
            - Silhouette optimal : k = {best_k_silhouette}
            
            **Recommandation :** Privilégier le score de silhouette → **k = {best_k_silhouette}**
            """)
            final_k = best_k_silhouette
        
          # Stocker pour les autres onglets
          st.session_state.optimal_k = final_k
          st.session_state.best_silhouette_score = best_score_silhouette
        
          st.success(f"✅ **k optimal déterminé : {final_k}** (stocké pour les étapes suivantes)")
    
    
        with tab3:
          st.subheader("Application du Clustering K-means")
        
        
          # Utiliser le k déterminé
          if 'optimal_k' in st.session_state:
             optimal_k = st.session_state.optimal_k
          else:
             optimal_k = 6  # Valeur par défaut basée sur votre analyse
        
          st.write(f"**Nombre de clusters sélectionné : k = {optimal_k}**")
        
          # Permettre l'ajustement
          n_clusters_selected = st.slider(
             "Ajustez k si nécessaire :", 
             min_value=2, 
             max_value=15, 
             value=optimal_k,
             key="k_slider"
           )
        
           # Appliquer K-means avec le k sélectionné
          with st.spinner(f"Application du clustering avec k={n_clusters_selected}..."):
            model = KMeans(n_clusters=n_clusters_selected, **KMEANS_PARAMS)
            model_kmeans = model.fit(base_scaled)
            labels = model_kmeans.labels_
            base['cluster'] = labels
        
          # Bouton
          apply_text = f"🚀 Appliquer K-means avec k={n_clusters_selected}"
          if 'recommended_k' in st.session_state and n_clusters_selected == st.session_state.recommended_k:
             apply_text += " (recommandé)"
        
          if st.button(apply_text, type="primary", key="apply_kmeans"):
             with st.spinner("Clustering en cours..."):
                model = KMeans(
                    n_clusters=n_clusters_selected, 
                    init='k-means++', 
                    max_iter=300, 
                    n_init=20,
                    random_state=42
                )
                model_kmeans = model.fit(base_scaled)
                labels = model_kmeans.labels_
                base['cluster'] = labels
                
                # Calculer le score silhouette
                if len(np.unique(labels)) > 1:
                    silhouette_avg = silhouette_score(base_scaled, labels)
                else:
                    silhouette_avg = 0
                
                # Sauvegarde
                st.session_state.kmeans_results = {
                    'base': base,
                    'labels': labels,
                    'model': model_kmeans,
                    'n_clusters': n_clusters_selected,
                    'silhouette_score': silhouette_avg,
                    'centers': model_kmeans.cluster_centers_
                }
                
                st.success(f"✅ Clustering terminé ! {n_clusters_selected} clusters créés.")
                st.info(f"📊 Score Silhouette final : {silhouette_avg:.3f}")
        
                 # Visualisation
                if 'kmeans_results' in st.session_state:
                  base_result = st.session_state.kmeans_results['base']
                  n_clusters_used = st.session_state.kmeans_results['n_clusters']
            
                  st.subheader("Visualisation des Clusters")
            
                  colonnes = ['Montant', 'Recence', 'Frequence', 'Quantite_totale']
                  col1, col2 = st.columns(2)
            
                with col1:
                   abscisses = st.selectbox("Variable pour l'axe X", colonnes, index=0, key="x_axis")
                with col2:
                  ordonnees = st.selectbox("Variable pour l'axe Y", colonnes, index=1, key="y_axis")
            
                  fig, ax = plt.subplots(figsize=(10, 6))
                  scatter = ax.scatter(base_result[abscisses], base_result[ordonnees], 
                                c=base_result['cluster'], cmap='Set1', s=50, alpha=0.7)
                  ax.set_xlabel(abscisses, fontsize=12)
                  ax.set_ylabel(ordonnees, fontsize=12)
                  ax.set_title(f'Clusters K-means (k={n_clusters_used})', fontsize=14, fontweight='bold')
                  ax.grid(True, alpha=0.3)
                  legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                  ax.add_artist(legend1)
                  st.pyplot(fig)

                  # Analyse de stabilité simplifiée
                  st.subheader("📈 Analyse de Stabilité")
            
                if st.button("Analyser la stabilité", type="secondary", key="analyze_stability"):
                  with st.spinner("Analyse de stabilité en cours..."):
                    try:
                     # Exécuter K-means plusieurs fois avec différentes initialisations
                      n_init = 10
                      ari_scores = []

                      for _ in range(n_init):
                        kmeans = KMeans(n_clusters=n_clusters_used, random_state=None)
                        labels = kmeans.fit_predict(base_scaled)
                        ari_scores.append(labels)

                        similarities = []
                        for i in range(len(ari_scores)-1):
                          similarities.append(
                          adjusted_rand_score(ari_scores[i], ari_scores[i+1])
                )

                        stability = np.mean(similarities)

            
                        st.write(f"**Stabilité de K-means :** {stability:.3f}")
                        if stability > 0.8:
                          st.success("✅ Excellente stabilité")
                        elif stability > 0.6:
                          st.info("📊 Bonne stabilité")
                        elif stability > 0.4:
                          st.warning("⚠️ Stabilité modérée")
                        else:
                          st.error("❌ Faible stabilité")
            
                         # Graphique de la stabilité
                        fig_stab, ax_stab = plt.subplots(figsize=(10, 4))
                        ax_stab.plot(range(1, n_init + 1), np.diag(similarities), 'o-', color='#667eea')
                        ax_stab.set_xlabel('Initialisation')
                        ax_stab.set_ylabel('Score ARI')
                        ax_stab.set_title(f'Stabilité pour k={n_clusters_used}')
                        ax_stab.grid(alpha=0.3)
                        st.pyplot(fig_stab)
            
                    except Exception as e:
                      st.error(f"Erreur lors de l'analyse de stabilité : {str(e)}")

                  # Contrat de maintenance
                      st.subheader("Contrat de maintenance")
                    try:
                      ARI_score = pd.read_csv("ARI_kmeans.csv")
                      st.write("Score ARI pour chaque semaine :")
                      st.dataframe(ARI_score)
                      ARI_scores = ARI_score['ARI'].tolist()

                     # Paramètres de style
                      sns.set(rc={'figure.figsize': (10, 6)})

                      # Création de la figure
                      fig, ax = plt.subplots()

                      # Tracé de la courbe
                      sns.lineplot(x=range(1, len(ARI_scores) + 1), y=ARI_scores, ax=ax)
                      ax.axvline(6, c='red', ls='--')

                      # Personnalisation
                      ax.set_title('Évolution du score ARI')
                      ax.set_xlabel('Semaines')
                      ax.set_ylabel('Score ARI')
                      ax.set_xticks(range(1, len(ARI_scores) + 1))
                      # Affichage dans Streamlit
                      st.pyplot(fig)
    
                    except FileNotFoundError:
                      st.  warning("⚠️ Fichier ARI_kmeans.csv non trouvé. Affichage d'un exemple.")
    
                     # Créer des données d'exemple
                      example_scores = [0.95, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83]
    
                      fig, ax = plt.subplots(figsize=(10, 6))
                      ax.plot(range(1, len(example_scores) + 1), example_scores, 'o-', color='#667eea')
                      ax.axvline(6, c='red', ls='--', label='Maintenance prévue')
                      ax.set_title('Exemple: Évolution du score ARI')
                      ax.set_xlabel('Semaines')
                      ax.set_ylabel('Score ARI')
                      ax.set_xticks(range(1, len(example_scores) + 1))
                      ax.grid(alpha=0.3)
                      ax.legend()
                      st.pyplot(fig)

                     # Recommandations de maintenance
                      st.markdown("#### 🛠️ Plan de maintenance recommandé")
        
                      maintenance_plan = [
                      "**🎯 Surveillance hebdomadaire** : Calculer le score ARI chaque semaine",
                      "**📊 Suivi du score silhouette** : Vérifier la qualité des clusters mensuellement",
                      "**🔄 Réentraînement** : Recalculer les clusters tous les 3 mois ou après 1000 nouveaux clients",
                      "**🚨 Alertes** : Alerter si score ARI < 0.7 ou score silhouette < 0.5",
                      "**📈 Revue trimestrielle** : Analyser l'évolution des segments avec l'équipe marketing",
                      "**🔧 Ajustements** : Réévaluer k si score silhouette baisse significativement"
                    ]
        
                for item in maintenance_plan:
                   st.markdown(f"- {item}")
        
                  # Export des résultats
                   st.markdown("#### 📥 Export des résultats")
        
                   if st.button("💾 Exporter les résultats du clustering"):
                  # Créer un DataFrame avec les résultats
                     results_df = base.copy()
                     results_df['cluster'] = results_df['cluster'].astype(str)
            
                     # Convertir en CSV
                csv = results_df.to_csv(index=False)
            
                st.download_button(
                 label="📄 Télécharger les résultats (CSV)",
                 data=csv,
                 file_name=f"clustering_k{n_clusters_selected}.csv",
                 mime="text/csv"
               )
    
        with tab4:
          st.subheader("Résultats et Interprétation")
        
          if 'kmeans_results' in st.session_state:
             base_result = st.session_state.kmeans_results['base']
             silhouette_score_val = st.session_state.kmeans_results.get('silhouette_score', 0)
             n_clusters_used = st.session_state.kmeans_results['n_clusters']
            
             # Métriques
             col1, col2, col3 = st.columns(3)
             with col1:
                 st.metric("Nombre de Clusters", n_clusters_used)
             with col2:
                 st.metric("Score Silhouette", f"{silhouette_score_val:.3f}")
             with col3:
                st.metric("Clients Clustérisés", len(base_result))
            
              # Statistiques
             st.subheader("📊 Statistiques par Cluster")
             cluster_stats = base_result.groupby('cluster').agg({
                'Recence': ['mean', 'std', 'min', 'max'],
                'Frequence': ['mean', 'std', 'min', 'max'],
                'Montant': ['mean', 'sum', 'count'],
                'Quantite_totale': ['mean', 'sum']
             }).round(2)
            
             cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
             safe_dataframe(cluster_stats)
            
             # Profilage
             st.subheader("🎯 Profilage des Clusters")
            
             for cluster_num in sorted(base_result['cluster'].unique()):
                cluster_data = base_result[base_result['cluster'] == cluster_num]
                n_clients = len(cluster_data)
                percentage = n_clients / len(base_result) * 100
                
                with st.expander(f"Cluster {cluster_num} - {n_clients} clients ({percentage:.1f}%)"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        recence_moy = cluster_data['Recence'].mean()
                        st.metric("Récence Moyenne", f"{recence_moy:.1f} jours")
                    with col2:
                        freq_moy = cluster_data['Frequence'].mean()
                        st.metric("Fréquence Moyenne", f"{freq_moy:.1f}")
                    with col3:
                        ca_moy = cluster_data['Montant'].mean()
                        st.metric("CA Moyen", f"€{ca_moy:,.0f}")
                    with col4:
                        qt_moy = cluster_data['Quantite_totale'].mean()
                        st.metric("Quantité Moyenne", f"{qt_moy:.0f}")
                    
                    # Recommandations
                    st.markdown("**💡 Recommandations Marketing :**")
                    
                    recence_moy = cluster_data['Recence'].mean()
                    freq_moy = cluster_data['Frequence'].mean()
                    ca_moy = cluster_data['Montant'].mean()
                    ca_total_moy = base_result['Montant'].mean()
                    
                    if recence_moy < 30 and freq_moy > 8:
                        st.markdown("- **Clients très actifs** : Programmes VIP, avantages exclusifs")
                    elif recence_moy > 90:
                        st.markdown("- **Clients inactifs** : Campagnes de réactivation, offres spéciales")
                    elif ca_moy > ca_total_moy * 1.5:
                        st.markdown("- **Clients à haute valeur** : Service personnalisé, offres premium")
                    elif freq_moy > 5 and ca_moy < ca_total_moy * 0.7:
                        st.markdown("- **Clients fréquents à faible valeur** : Promotions volume, fidélité")
                    else:
                        st.markdown("- **Clients réguliers** : Maintien de la relation, newsletters")
    
    # SEGMENTATION RFM 
    def segmentation_rfm_func(df_for_rfm):
        st.markdown("""
        <div style='background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #1E3A8A; margin: 0;'>⭐ Segmentation RFM des Clients</h3>
            <p style='margin: 5px 0 0 0;'>Analyse de la Recence, Fréquence et Montant pour segmenter les clients.</p>
        </div>
        """, unsafe_allow_html=True)

         # Introduction
        with st.expander("ℹ️ À propos de RFM", expanded=False):
         st.markdown("""
        **RFM** (Recency, Frequency, Monetary) segmente les clients selon :
        
        - **📅 Récence (R)** : Délai depuis le dernier achat
        - **🔄 Fréquence (F)** : Nombre d'achats
        - **💰 Monétaire (M)** : Montant total dépensé
        
        **🎯 Objectifs :**
        - Identifier les meilleurs clients
        - Cibler les campagnes marketing
        - Personnaliser les communications
        - Optimiser la rétention
        """)
        
        # Calcul RFM
        rfm = df_for_rfm.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,
            'InvoiceNo': 'count',
            'Montant': 'sum'
        }).rename(columns={
            'InvoiceDate': 'Recence',
            'InvoiceNo': 'Frequence',
            'Montant': 'Montant'
        })
        
        # Calcul des quartiles
        quartiles = rfm[['Recence', 'Frequence', 'Montant']].quantile([0.25, 0.5, 0.75]).to_dict()
        
        # Fonctions de scoring
        def r_score(x):
            if x <= quartiles['Recence'][0.25]:
                return 4
            elif quartiles['Recence'][0.25] < x <= quartiles['Recence'][0.5]:
                return 3
            elif quartiles['Recence'][0.5] < x <= quartiles['Recence'][0.75]:
                return 2
            else:
                return 1
        
        def fm_score(x, col):
            if x <= quartiles[col][0.25]:
                return 1
            elif quartiles[col][0.25] < x <= quartiles[col][0.5]:
                return 2
            elif quartiles[col][0.5] < x <= quartiles[col][0.75]:
                return 3
            else:
                return 4
        
        # Application des scores
        rfm['R_Score'] = rfm['Recence'].apply(lambda x: r_score(x))
        rfm['F_Score'] = rfm['Frequence'].apply(lambda x: fm_score(x, 'Frequence'))
        rfm['M_Score'] = rfm['Montant'].apply(lambda x: fm_score(x, 'Montant'))
        rfm['RFM_score'] = rfm['R_Score'].map(str) + rfm['F_Score'].map(str) + rfm['M_Score'].map(str)
        
        # Segmentation
        code_segt = {
            r'11': 'Clients en hibernation',
            r'1[2-3]': 'Clients à risque',
            r'14': 'Clients à ne pas perdre',
            r'21': 'Clients presqu\'endormis',
            r'22': 'Clients à suivre',
            r'[2-3][3-4]': 'Clients loyaux',
            r'31': 'Clients prometteurs',
            r'41': 'Nouveaux clients',
            r'[3-4]2': 'Clients potentiellement loyaux',
            r'4[3-4]': 'Très bons clients'
        }
        
        rfm['Segment'] = rfm['R_Score'].map(str) + rfm['F_Score'].map(str)
        rfm['Segment'] = rfm['Segment'].replace(code_segt, regex=True)
        
        # Onglets pour l'analyse RFM
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores RFM", "👥 Segmentation", "🎯 Analyse", "💾 Export"])
        
        with tab1:
            st.subheader("Scores RFM des Clients")
            
            # Métriques RFM globales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recence Moyenne", f"{rfm['Recence'].mean():.1f} jours")
            with col2:
                st.metric("Fréquence Moyenne", f"{rfm['Frequence'].mean():.1f}")
            with col3:
                st.metric("Montant Moyen", f"€{rfm['Montant'].mean():,.0f}")
            
            # Distribution des scores - REMPLACER LE GRAPHIQUE CIRCULAIRE PAR UN BARRES
            st.subheader("Distribution des Scores RFM")
            
            # Créer un dataframe pour la visualisation
            scores_data = []
            for score_type in ['R_Score', 'F_Score', 'M_Score']:
                score_counts = rfm[score_type].value_counts().sort_index()
                for score, count in score_counts.items():
                    scores_data.append({
                        'Score Type': score_type.replace('_Score', ''),
                        'Score': score,
                        'Nombre de Clients': count
                    })
            
            scores_df = pd.DataFrame(scores_data)
            
            # Graphique à barres groupées
            fig = px.bar(scores_df, x='Score', y='Nombre de Clients', color='Score Type',
                        barmode='group', title='Distribution des Scores RFM',
                        color_discrete_sequence=['#667eea', '#764ba2', '#4CAF50'])
            
            fig.update_layout(
                xaxis_title="Score",
                yaxis_title="Nombre de Clients",
                legend_title="Type de Score"
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Affichage des données RFM
            safe_dataframe(rfm.head(20))
        
        with tab2:
            st.subheader("Segmentation des Clients")
            
            # Répartition des segments
            segments_counts = rfm['Segment'].value_counts()
            
            # REMPLACER LE GRAPHIQUE CIRCULAIRE PAR UN BARRES HORIZONTAL
            st.subheader("📊 Répartition des Clients par Segment")
            
            # Créer un dataframe pour la visualisation
            segments_df = segments_counts.reset_index()
            segments_df.columns = ['Segment', 'Nombre de Clients']
            segments_df = segments_df.sort_values('Nombre de Clients', ascending=True)
            
            # Graphique à barres horizontales
            fig = px.bar(segments_df, y='Segment', x='Nombre de Clients',
                        orientation='h', title='Répartition des Segments Clients',
                        color='Nombre de Clients', color_continuous_scale='Blues',
                        text='Nombre de Clients')
            
            fig.update_layout(
                yaxis_title="Segment",
                xaxis_title="Nombre de Clients",
                showlegend=False
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Tableau des segments
            st.subheader("📋 Tableau des Segments")
            segments_table = segments_counts.reset_index()
            segments_table.columns = ['Segment', 'Nombre de Clients']
            segments_table['% Total'] = (segments_table['Nombre de Clients'] / len(rfm) * 100).round(1)
            segments_table = segments_table.sort_values('Nombre de Clients', ascending=False)
            
            safe_dataframe(segments_table)
        
        with tab3:
            st.subheader("Analyse des Segments")
            
            # Statistiques par segment
            segment_stats = rfm.groupby('Segment').agg({
                'Recence': 'mean',
                'Frequence': 'mean',
                'Montant': ['mean', 'sum', 'count']
            }).round(2)
            
            # Aplatir les colonnes multi-index
            segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns.values]
            segment_stats = segment_stats.reset_index()
            
            safe_dataframe(segment_stats)
            
            # Recommandations par segment
            st.subheader("🎯 Stratégies Marketing par Segment")
            
            strategies = {
                'Très bons clients': [
                    "Programme VIP avec avantages exclusifs",
                    "Accès anticipé aux nouveautés",
                    "Service client prioritaire",
                    "Invitations à des événements privés"
                ],
                'Clients loyaux': [
                    "Programme de fidélité avec points",
                    "Offres personnalisées régulières",
                    "Sondages pour amélioration du service",
                    "Recommend-a-friend bonus"
                ],
                'Clients à ne pas perdre': [
                    "Offres de réactivation ciblées",
                    "Enquêtes de satisfaction",
                    "Support client proactif",
                    "Programmes de parrainage"
                ],
                'Nouveaux clients': [
                    "Email de bienvenue avec cadeau",
                    "Guide du débutant",
                    "Première commande avec réduction",
                    "Tutoriels produits"
                ],
                'Clients à risque': [
                    "Campagnes de rétention",
                    "Offres spéciales de retour",
                    "Feedback sur les raisons de départ",
                    "Programmes de récompenses"
                ]
            }
            
            for segment, actions in strategies.items():
                if segment in rfm['Segment'].unique():
                    segment_count = len(rfm[rfm['Segment'] == segment])
                    segment_percentage = (segment_count / len(rfm)) * 100
                    
                    with st.expander(f"{segment} ({segment_count} clients, {segment_percentage:.1f}%)"):
                        for i, action in enumerate(actions, 1):
                            st.markdown(f"{i}. {action}")
        
        with tab4:
            st.subheader("Export des Résultats RFM")
            
            # Options d'export
            export_format = st.radio("Format d'export", ["CSV", "Excel"], horizontal=True)
            
            if export_format == "CSV":
                csv = rfm.reset_index().to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les segments RFM (CSV)",
                    data=csv,
                    file_name="segmentation_rfm.csv",
                    mime="text/csv",
                    width="stretch"
                )
            else:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    rfm.reset_index().to_excel(writer, index=False, sheet_name='Segments_RFM')
                st.download_button(
                    label="📥 Télécharger les segments RFM (Excel)",
                    data=output.getvalue(),
                    file_name="segmentation_rfm.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch"
                )
            
            # Résumé exécutif
            st.subheader("📋 Résumé Exécutif")
            
            top_segment = segments_counts.index[0]
            top_count = segments_counts.iloc[0]
            top_percentage = (top_count / len(rfm)) * 100
            
            st.markdown(f"""
            **Insights Clés :**
            
            - **Segment majoritaire :** {top_segment} ({top_percentage:.1f}% des clients)
            - **Clients à haute valeur :** {len(rfm[rfm['M_Score'] == 4])} clients (score M=4)
            - **Clients récents :** {len(rfm[rfm['R_Score'] == 4])} clients (score R=4)
            - **Clients fréquents :** {len(rfm[rfm['F_Score'] == 4])} clients (score F=4)
            - **Clients champions :** {len(rfm[rfm['RFM_score'] == '444'])} clients (score RFM=444)
            
            **Recommandations Globales :**
            1. Focus sur la rétention des {len(rfm[rfm['Segment'].isin(['Très bons clients', 'Clients loyaux'])])} clients loyaux
            2. Campagnes de réactivation pour les {len(rfm[rfm['Segment'] == 'Clients à risque'])} clients à risque
            3. Programmes d'onboarding pour les {len(rfm[rfm['Segment'] == 'Nouveaux clients'])} nouveaux clients
            """)
    
    # FP-GROWTH 
    def fp_growth_func(df_for_fp_growth):
        st.markdown("""
        <div style='background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #1E3A8A; margin: 0;'>🛒 Analyse des Associations avec FP-Growth</h3>
            <p style='margin: 5px 0 0 0;'>Découvrez les associations entre produits pour des recommandations intelligentes.</p>
        </div>
        """, unsafe_allow_html=True)

        # Introduction
        with st.expander("ℹ️ À propos de FP-Growth", expanded=False):
         st.markdown("""
        **FP-Growth** (Frequent Pattern Growth) est un algorithme efficace pour découvrir des règles d'association.
        
        **📊 Métriques clés :**
        - **Support** : Fréquence de l'itemset dans les transactions
        - **Confiance** : Probabilité que B soit acheté si A est acheté  
        - **Lift** : Mesure de l'intérêt de la règle (>1 = association positive)
        
        **⚡ Avantages vs Apriori :**
        - Une seule passe sur la base de données
        - Structure FP-tree compacte
        - Meilleure performance sur données denses
        """)
        
        # Chargement des règles d'association
        try:
            @st.cache_data
            def load_rules():
                rules = pd.read_csv("regle_association.csv")
                rules['antecedents'] = rules['antecedents'].apply(ast.literal_eval)
                rules['consequents'] = rules['consequents'].apply(ast.literal_eval)
                return rules
            rules = load_rules()
            st.success(f"✅ {len(rules)} règles d'association chargées")
        except FileNotFoundError:
            st.warning("⚠️ Fichier `regle_association.csv` non trouvé. Utilisation de données de démonstration.")
            # Création de règles d'exemple
            rules_data = {
                'antecedents': [['WHITE HANGING HEART T-LIGHT HOLDER'], 
                               ['REGENCY CAKESTAND 3 TIER'],
                               ['JUMBO BAG RED RETROSPOT']],
                'consequents': [['WHITE METAL LANTERN'], 
                               ['GARDENERS KNEELING PAD CORDLESS'],
                               ['LUNCH BAG RED RETROSPOT']],
                'support': [0.05, 0.03, 0.04],
                'confidence': [0.85, 0.78, 0.92],
                'lift': [2.5, 3.1, 4.2]
            }
            rules = pd.DataFrame(rules_data)
            rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: frozenset(x))
        
        # Onglets pour FP-Growth
        tab1, tab2, tab3 = st.tabs(["🔍 Recommandations", "📊 Statistiques", "⚙️ Paramètres"])
        
        with tab1:
            st.subheader("Système de Recommandation en Temps Réel")
            
            # Interface de sélection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Extraction des items uniques
                unique_items = sorted(set(
                    item for ant in rules['antecedents'] for item in ant
                ).union(
                    item for con in rules['consequents'] for item in con
                ))
                
                if unique_items:
                    selected_item = st.selectbox(
                        "Sélectionnez un produit :",
                        unique_items,
                        help="Choisissez un produit pour voir les recommandations associées"
                    )
                else:
                    st.error("Aucun produit trouvé dans les règles d'association")
                    return
            
            with col2:
                st.markdown("### Paramètres")
                min_conf = st.slider(
                    "Confiance minimale",
                    0.0, 1.0, 0.5, 0.05,
                    help="Probabilité que le produit recommandé soit acheté avec le produit sélectionné"
                )
                min_lift = st.slider(
                    "Lift minimal",
                    0.5, 5.0, 1.2, 0.1,
                    help="Mesure de l'intérêt de la règle (1=indépendant, >1=association positive)"
                )
            
            # Bouton de génération
            if st.button("🎯 Générer les Recommandations", type="primary"):
                with st.spinner("Analyse des associations..."):
                    # Filtrage des règles
                    filtered_rules = rules[
                        rules['antecedents'].apply(lambda x: selected_item in x) &
                        (rules['confidence'] >= min_conf) &
                        (rules['lift'] >= min_lift)
                    ].sort_values(by='confidence', ascending=False)
                    
                    # Suppression des doublons
                    filtered_rules = filtered_rules.drop_duplicates(subset='consequents')
                    
                    # Affichage des résultats
                    if not filtered_rules.empty:
                        st.success(f"✅ {len(filtered_rules)} recommandations trouvées")
                        
                        # Métriques
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confiance Moyenne", f"{filtered_rules['confidence'].mean():.2%}")
                        with col2:
                            st.metric("Lift Moyen", f"{filtered_rules['lift'].mean():.2f}")
                        with col3:
                            st.metric("Support Moyen", f"{filtered_rules['support'].mean():.2%}")
                        
                        # Affichage des recommandations
                        st.subheader("📦 Produits Recommandés")
                        
                        for idx, (_, row) in enumerate(filtered_rules.iterrows(), 1):
                            recommended_items = ', '.join(list(row['consequents']))
                            
                            with st.container():
                                st.markdown(f"""
                                <div style='background: white; padding: 15px; border-radius: 10px; 
                                            border-left: 5px solid #667eea; margin: 10px 0; 
                                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <div>
                                            <h4 style='margin: 0; color: #1E3A8A;'>Recommandation #{idx}</h4>
                                            <p style='font-size: 1.1rem; color: #333; font-weight: bold; margin: 5px 0;'>
                                                {recommended_items}
                                            </p>
                                            <small style='color: #666;'>
                                                Si vous achetez <strong>{selected_item}</strong>, vous pourriez aussi aimer :
                                            </small>
                                        </div>
                                        <div style='text-align: right;'>
                                            <div style='display: flex; gap: 15px;'>
                                                <div>
                                                    <small style='color: #666;'>Confiance</small>
                                                    <div style='background: #e3f2fd; padding: 5px 10px; 
                                                                border-radius: 5px; font-weight: bold; color: #1976d2;'>
                                                        {row['confidence']:.1%}
                                                    </div>
                                                </div>
                                                <div>
                                                    <small style='color: #666;'>Lift</small>
                                                    <div style='background: #f3e5f5; padding: 5px 10px; 
                                                                border-radius: 5px; font-weight: bold; color: #7b1fa2;'>
                                                        {row['lift']:.2f}
                                                    </div>
                                                </div>
                                                <div>
                                                    <small style='color: #666;'>Support</small>
                                                    <div style='background: #e8f5e8; padding: 5px 10px; 
                                                                border-radius: 5px; font-weight: bold; color: #388e3c;'>
                                                        {row['support']:.1%}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Graphique des règles
                        st.subheader("📈 Visualisation des Associations")
                        
                        top_rules = filtered_rules.head(10)
                        fig = px.scatter(top_rules, x='confidence', y='lift',
                                        size='support', color='lift',
                                        hover_name=top_rules['consequents'].apply(lambda x: ', '.join(list(x))),
                                        title='Top 10 des Associations',
                                        labels={'confidence': 'Confiance', 'lift': 'Lift', 'support': 'Support'},
                                        color_continuous_scale='Viridis')
                        
                        st.plotly_chart(fig, width="stretch")
                    
                    else:
                        st.warning(f"⚠️ Aucune recommandation trouvée pour '{selected_item}' avec les critères spécifiés.")
                        st.info("💡 Essayez de réduire les seuils de confiance ou de lift.")
        
        with tab2:
            st.subheader("Statistiques des Règles d'Association")
            
            # Métriques globales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total des Règles", len(rules))
            with col2:
                st.metric("Confiance Moyenne", f"{rules['confidence'].mean():.2%}")
            with col3:
                st.metric("Lift Moyen", f"{rules['lift'].mean():.2f}")
            with col4:
                st.metric("Support Moyen", f"{rules['support'].mean():.2%}")
            
            # Distribution des confidences
            st.subheader("Distribution des Niveaux de Confiance")
            fig1 = px.histogram(rules, x='confidence', nbins=20,
                               title='Distribution des Confiances',
                               labels={'confidence': 'Confiance', 'count': 'Nombre de Règles'},
                               color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig1, width="stretch")
            
            # Top 10 règles par lift
            st.subheader("🏆 Top 10 des Meilleures Associations")
            top_rules = rules.nlargest(10, 'lift')
            
            # Formatage pour l'affichage
            display_rules = top_rules.copy()
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            safe_dataframe(display_rules[['antecedents', 'consequents', 'confidence', 'lift', 'support']],
                        )
            
            # Graphique des top règles
            fig2 = px.bar(top_rules, x=top_rules.index, y='lift',
                         title='Top 10 Règles par Lift',
                         labels={'index': 'Règle', 'lift': 'Lift'},
                         color='confidence',
                         color_continuous_scale='Viridis')
            
            fig2.update_layout(xaxis=dict(tickmode='array',
                                         tickvals=list(range(10)),
                                         ticktext=[f"Règle {i+1}" for i in range(10)]))
            
            st.plotly_chart(fig2, width="stretch")
        
        with tab3:
            st.subheader("Paramètres Avancés de l'Analyse")
            
            with st.form("parametres_avances"):
                st.markdown("### ⚙️ Configuration de l'Analyse")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    min_support = st.number_input("Support minimum (%)", 
                                                 min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    max_items = st.number_input("Nombre maximum d'items par règle", 
                                               min_value=2, max_value=5, value=3)
                
                with col2:
                    include_reverse = st.checkbox("Inclure les règles inverses", value=True)
                    filter_redundant = st.checkbox("Filtrer les règles redondantes", value=True)
                
                st.markdown("### 📊 Métriques de Performance")
                
                performance_metrics = st.multiselect(
                    "Métriques à calculer",
                    ["Confidence", "Lift", "Conviction", "Leverage"],
                    default=["Confidence", "Lift"]
                )
                
                if st.form_submit_button("🔄 Appliquer les Paramètres", type="primary"):
                    st.info("Les paramètres seront appliqués lors de la prochaine analyse FP-Growth.")
                    st.session_state.fp_params = {
                        'min_support': min_support,
                        'max_items': max_items,
                        'include_reverse': include_reverse,
                        'filter_redundant': filter_redundant,
                        'metrics': performance_metrics
                    }
            
            # Informations sur l'algorithme
            with st.expander("ℹ️ À propos de FP-Growth"):
                st.markdown("""
                **FP-Growth (Frequent Pattern Growth)** est un algorithme efficace pour la découverte de règles d'association.
                
                **Avantages :**
                - Plus rapide que Apriori pour les grands jeux de données
                - Ne nécessite pas de génération de candidats multiples
                - Utilise une structure d'arbre compacte (FP-Tree)
                
                **Métriques :**
                - **Support :** Fréquence de l'itemset dans les transactions
                - **Confidence :** Probabilité conditionnelle P(B|A)
                - **Lift :** Mesure de l'intérêt de la règle (indépendance = 1)
                
                **Applications :**
                - Recommandations de produits
                - Analyse de panier d'achat
                - Marketing cross-selling
                """)
    
    # Exécution de la méthode sélectionnée
    if choix == "👥 K-means":
        kmeans_clustering(df_invoice, analysis_date)
    elif choix == "⭐ Segmentation RFM":
        segmentation_rfm_func(df_invoice)
    else:  # FP_GROWTH
        fp_growth_func(df)

# ========
# RÉSUMÉ 
# ========
def Summary():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 30px;'>
        <h2 style='margin: 0;'>📈 Résumé Exécutif des Ventes</h2>
        <p style='opacity: 0.9;'>Vue d'ensemble des performances et insights clés</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df_selection = st.session_state['df'].copy()
        
        # Assurez-vous que 'InvoiceDate' est datetime
        df_selection['InvoiceDate'] = pd.to_datetime(df_selection['InvoiceDate'])
        
        # Calcul des métriques principales
        total_sales = df_selection["Montant"].sum()
        num_unique_customers = df_selection['CustomerID'].dropna().nunique()
        avg_sales_per_transaction = round(df_selection.groupby('InvoiceNo')['Montant'].sum().mean(), 2)
        total_transactions = df_selection['InvoiceNo'].nunique()
        
        # Affichage des métriques avec style
        st.markdown("### 📊 Indicateurs Clés de Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin: 0; font-size: 1.5rem;'>Chiffre d'Affaires Total</h3>
                <p style='font-size: 2rem; font-weight: bold; margin: 10px 0;'>€{total_sales:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #2196F3 0%, #21CBF3 100%);'>
                <h3 style='margin: 0; font-size: 1.5rem;'>Clients Uniques</h3>
                <p style='font-size: 2rem; font-weight: bold; margin: 10px 0;'>{num_unique_customers:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);'>
                <h3 style='margin: 0; font-size: 1.5rem;'>Panier Moyen</h3>
                <p style='font-size: 2rem; font-weight: bold; margin: 10px 0;'>€{avg_sales_per_transaction:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);'>
                <h3 style='margin: 0; font-size: 1.5rem;'>Transactions</h3>
                <p style='font-size: 2rem; font-weight: bold; margin: 10px 0;'>{total_transactions:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analyse des produits
        st.markdown("### 🛒 Analyse des Produits")
        
        col_charts1, col_charts2 = st.columns(2)
        
        with col_charts1:
            # Top 5 produits par CA
            sales_by_description = df_selection.groupby(by=["Description"])[["Montant"]].sum().sort_values(
                by="Montant", ascending=False).head(5)
            
            fig_description_sales = px.bar(
                sales_by_description,
                y="Montant",
                x=sales_by_description.index,
                orientation="v",
                title="<b>Top 5 Produits par Chiffre d'Affaires</b>",
                color=sales_by_description["Montant"],
                color_continuous_scale="Viridis",
                template="plotly_white",
            )
            
            fig_description_sales.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, tickangle=45),
                yaxis_title="Chiffre d'Affaires (€)"
            )
            
            st.plotly_chart(fig_description_sales, width="stretch")
        
        with col_charts2:
        # Top 5 pays par CA
         sales_by_country = df_selection.groupby(by=["Country"])[["Montant"]].sum().sort_values(
         by="Montant", ascending=False).head(5)
    
        #Utiliser une échelle de couleurs séquentielle valide
         fig_country_sales = px.bar(
         sales_by_country.reset_index(),
         x="Country",
         y="Montant",
         title="<b>Top 5 Pays par Chiffre d'Affaires</b>",
         color="Montant",
         color_continuous_scale="Viridis",  # Changé de 'Set3' à 'Viridis'
         text="Montant"
        )
    
        fig_country_sales.update_layout(
        xaxis_title="Pays",
        yaxis_title="Chiffre d'Affaires (€)",
        xaxis_tickangle=45
        )
    
        st.plotly_chart(fig_country_sales, width="stretch")
        
        st.markdown("---")
        
        # Analyse temporelle
        st.markdown("### 📅 Analyse Temporelle")
        
        # Préparation des données temporelles
        df_selection['Mois'] = df_selection['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_sales = df_selection.groupby('Mois')['Montant'].sum().reset_index()
        
        fig_temporal = px.line(
            monthly_sales,
            x='Mois',
            y='Montant',
            title='Évolution Mensuelle du Chiffre d\'Affaires',
            markers=True,
            line_shape='spline'
        )
        
        fig_temporal.update_layout(
            xaxis_title="Mois",
            yaxis_title="Chiffre d'Affaires (€)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_temporal, width="stretch")
        
        
        st.markdown("### 👥 Segmentation RFM des Clients")

        # -------------------------------
        # Calcul RFM
        # -------------------------------
        # On utilise analysis_date comme le jour suivant la dernière transaction
        analysis_date = df_selection['InvoiceDate'].max() + timedelta(days=1)

        rfm_df = df_selection.groupby('CustomerID').agg(
          Recence=('InvoiceDate', lambda date: (analysis_date - date.max()).days),
          Frequence=('InvoiceNo', 'nunique'),
          Montant=('Montant', 'sum')
        ).reset_index()

         # Quartiles pour la segmentation
        quartiles = rfm_df[['Recence', 'Frequence', 'Montant']].quantile([0.25, 0.5, 0.75]).to_dict()

         # Fonctions de score R, F, M
        def r_score(x):
          if x <= quartiles['Recence'][0.25]:
            return 4
          elif quartiles['Recence'][0.25] < x <= quartiles['Recence'][0.5]:
            return 3
          elif quartiles['Recence'][0.5] < x <= quartiles['Recence'][0.75]:
            return 2
          else:
           return 1

        def fm_score(x, col):
         if x <= quartiles[col][0.25]:
          return 1
         elif quartiles[col][0.25] < x <= quartiles[col][0.5]:
          return 2
         elif quartiles[col][0.5] < x <= quartiles[col][0.75]:
          return 3
         else:
          return 4

         # Application des scores
        rfm_df['R_Score'] = rfm_df['Recence'].apply(lambda x: r_score(x))
        rfm_df['F_Score'] = rfm_df['Frequence'].apply(lambda x: fm_score(x, 'Frequence'))
        rfm_df['M_Score'] = rfm_df['Montant'].apply(lambda x: fm_score(x, 'Montant'))

         # -------------------------------
         # Segmentation RFM
         # -------------------------------
        code_segt = {
          r'11': 'Clients en hibernation',
          r'1[2-3]': 'Clients à risque',
          r'14': 'Clients à ne pas perdre',
          r'21': 'Clients presqu\'endormis',
          r'22': 'Clients à suivre',
          r'[2-3][3-4]': 'Clients loyaux',
          r'31': 'Clients prometteurs',
          r'41': 'Nouveaux clients',
          r'[3-4]2': 'Clients potentiellement loyaux',
          r'4[3-4]': 'Très bons clients'
       }

        def apply_rfm_segment(row):
         # Conversion en int pour éviter les floats
         r_f_score_str = f"{int(row['R_Score'])}{int(row['F_Score'])}"
         for pattern, segment_name in code_segt.items():
           if re.match(pattern, r_f_score_str):
            return segment_name
         return 'Non défini'

        rfm_df['Segment'] = rfm_df.apply(apply_rfm_segment, axis=1)

         # -------------------------------
         # Affichage du graphe des segments
         # -------------------------------
        st.subheader("📊 Répartition des Clients par Segment")
        fig, ax = plt.subplots(figsize=(12, 10))
        segments_counts = rfm_df['Segment'].value_counts().sort_values(ascending=True)
        norm = plt.Normalize(segments_counts.min(), segments_counts.max())
        colors = cm.Blues(norm(segments_counts.values))
        bars = ax.barh(range(len(segments_counts)), segments_counts, color=colors)
        ax.set_frame_on(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.set_yticks(range(len(segments_counts)))
        ax.set_yticklabels(segments_counts.index)

         # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
          value = bar.get_width()
          ax.text(value, bar.get_y() + bar.get_height() / 2,
            '{:,} ({:}%)'.format(int(value), int(value * 100 / segments_counts.sum())),
            va='center', ha='left')

        st.pyplot(fig)

        # -------------------------------
        # Top 5 clients par Montant
        # -------------------------------
        st.subheader("Meilleurs Clients (Top 5 par Total des Achats)")
        safe_dataframe(rfm_df.nlargest(5, 'Montant'))

        
        # Recommandations
        st.markdown("---")
        st.markdown("### 🎯 Recommandations Stratégiques")
        
        # Création de la liste avec condition intégrée
        recommendations = [
            "**Focus sur la rétention** : Les clients existants génèrent 80% du revenu récurrent",
            "**Personnalisation** : Utiliser les segments RFM pour des campagnes ciblées",
            *(["**Cross-selling** : Exploiter les associations de produits identifiées"] 
              if hasattr(st.session_state, 'fp_params') else []),
            "**Optimisation des prix** : Analyser l'élasticité-prix des top produits",
            "**Expansion géographique** : Cibler les pays à fort potentiel de croissance"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Bouton d'export du rapport
        st.markdown("---")
        st.markdown("### 💾 Export du Rapport")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export des données résumées
            summary_data = pd.DataFrame({
                'Métrique': ['CA Total', 'Clients Uniques', 'Panier Moyen', 'Transactions', 'Produits Uniques'],
                'Valeur': [
                    f"€{total_sales:,.0f}",
                    f"{num_unique_customers:,}",
                    f"€{avg_sales_per_transaction:,.2f}",
                    f"{total_transactions:,}",
                    f"{df_selection['Description'].nunique():,}"
                ]
            })
            
            csv_summary = summary_data.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger le Résumé (CSV)",
                data=csv_summary,
                file_name="resume_ventes.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col_export2:
            # Export des top clients
            csv_top_clients = rfm_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger Top Clients (CSV)",
                data=csv_top_clients,
                file_name="top_clients.csv",
                mime="text/csv",
                width="stretch"
            )
    
    else:
        st.warning("Veuillez d'abord importer un fichier de données pour afficher le résumé.")

# =======================
# PAGE "À PROPOS DE NOUS" 
# =======================
def about_us():
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 40px;'>
        <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700;'>📊 Analyse et Segmentation des Clients</h1>
        <p style='font-size: 1.3rem; opacity: 0.95; margin-top: 10px;'>Décision Marketing par Data Mining</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Image
    try:
        image = Image.open("e-commerce.jpg")
        st.image(image, caption="E-commerce Analytics", width="stretch")
    except:
        st.info("L'image 'e-commerce.jpg' n'a pas été trouvée. Assurez-vous qu'elle est dans le répertoire courant.")
    
    # Description
    st.markdown("""
    ### 🎯 Notre Mission
    
    Nous sommes une équipe dédiée à l'analyse des données clients pour améliorer les stratégies commerciales.
    Notre objectif est de fournir des insights actionnables à partir des données transactionnelles pour aider 
    les entreprises à mieux comprendre leurs clients et optimiser leurs opérations.
    
    ### 🔧 Nos Compétences
    
    - **Data Mining** : Extraction de patterns et d'associations
    - **Machine Learning** : Modélisation prédictive et clustering
    - **Visualisation** : Dashboards interactifs et rapports
    - **Business Intelligence** : Transformation des données en décisions
    
    ### 📈 Notre Approche
    
    1. **Compréhension** : Analyse approfondie des données et du contexte métier
    2. **Modélisation** : Application d'algorithmes avancés de data mining
    3. **Visualisation** : Création de dashboards interactifs et intuitifs
    4. **Action** : Recommandations concrètes pour l'amélioration des performances
    """)
    
    st.markdown("---")
    
    # Équipe
    st.markdown("### 👥 Notre Équipe")
    
    # Création des colonnes pour l'équipe
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: #1E3A8A;'>Ngone NDAO</h3>
            <h3 style='color: #1E3A8A;'>Dib NDIAYE</h3>
            <h3 style='color: #1E3A8A;'>Mareme DIONE</h3>
            <h3 style='color: #1E3A8A;'>Mouhammad SONKO</h3>
            <h3 style='color: #1E3A8A;'>Djimith NDAIAYE</h3>
            <h3 style='color: #1E3A8A;'>Ababacar MBENGUE</h3>
            <p style='color: #666;'>Data Scientist & Analyste</p>
            <div style='margin: 15px 0;'>
                <span style='background: #e3f2fd; color: #1976d2; padding: 5px 10px; border-radius: 5px; margin: 2px; display: inline-block;'>
                    Python
                </span>
                <span style='background: #e3f2fd; color: #1976d2; padding: 5px 10px; border-radius: 5px; margin: 2px; display: inline-block;'>
                    Machine Learning
                </span>
                <span style='background: #e3f2fd; color: #1976d2; padding: 5px 10px; border-radius: 5px; margin: 2px; display: inline-block;'>
                    Data Visualization
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='color: #1E3A8A;'>Étudiant SID</h3>
            <p style='color: #666;'>Master SID</p>
            <div style='margin: 15px 0;'>
                <span style='background: #f3e5f5; color: #7b1fa2; padding: 5px 10px; border-radius: 5px; margin: 2px; display: inline-block;'>
                    Statistiques
                </span>
                <span style='background: #f3e5f5; color: #7b1fa2; padding: 5px 10px; border-radius: 5px; margin: 2px; display: inline-block;'>
                    Big Data
                </span>
                <span style='background: #f3e5f5; color: #7b1fa2; padding: 5px 10px; border-radius: 5px; margin: 2px; display: inline-block;'>
                    Business Intelligence
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Contact
    st.markdown("### 📞 Contactez-nous")
    
    # Grille de contact
    contact_col1, contact_col2, contact_col3, contact_col4 = st.columns(4)
    
    with contact_col1:
        st.markdown("""
        <div style='text-align: center;'>
            <h4>👤 Nom</h4>
            <p><strong>Ngone NDAO</strong></p>
            <p><strong>Mouhammad SONKO</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with contact_col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h4>📧 Email</h4>
            <p><a href="mailto:ndaongone089@gmail.com" style='color: #667eea; text-decoration: none;'>
                ndaongone089@gmail.com
            </a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with contact_col3:
        st.markdown("""
        <div style='text-align: center;'>
            <h4>📱 Contact</h4>
            <p><a href="tel:+221763962838" style='color: #667eea; text-decoration: none;'>
                +221 763962838
            </a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with contact_col4:
        st.markdown("""
        <div style='text-align: center;'>
            <h4>💼 LinkedIn</h4>
            <p><a href="https://www.linkedin.com/in/ngon%C3%A9-ndao-163162305/" 
                  target="_blank" style='color: #667eea; text-decoration: none;'>
                Profil LinkedIn
            </a></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>© 2026 Projet Data Mining - Master SID</p>
        <p><small>Développé avec ❤️ et Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# PIED DE PAGE GLOBAL
# ===============================
def global_footer():
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 2])
    
    with footer_col1:
        st.markdown("""
        **Projet Data Mining 2026**  
        Analyse et segmentation des clients e-commerce  
        Alioune Diop de Bambey - Master Statistique et Informatique Deisionnelle
        """)
    
    with footer_col2:
        st.markdown("""
        **Version** : 2.0.0  
        **Dernière mise à jour** : 2026
        """)
    
    with footer_col3:
        st.markdown("""
        **Contact** : [ndaongone089@gmail.com](mailto:ndaongone089@gmail.com)  
        **GitHub** : [Repository du Projet](https://github.com/)
        """)
    
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666; font-size: 0.9rem;'>
        © 2026 Data Mining Dashboard. Tous droits réservés.
    </div>
    """, unsafe_allow_html=True)

# ===============================
# APPLICATION PRINCIPALE
# ===============================
def main():
    # Créer la sidebar
    create_sidebar()
    
    # Routage basé sur le choix du menu
    if st.session_state.menu_choice == "🏠 Accueil":
        home_page()
    elif st.session_state.menu_choice == "📋 Description":
        description_data()
    elif st.session_state.menu_choice == "📊 Visualisation":
        visualize_data()
    elif st.session_state.menu_choice == "🤖 Modélisation":
        modeling_and_predictions()
    elif st.session_state.menu_choice == "📈 Résumé":
        Summary()
    elif st.session_state.menu_choice == "👥 À propos de nous":
        about_us()
    else:
        about_us()  # Page par défaut
    
    # Ajouter le pied de page global
    global_footer()

# ===============================
# LANCEMENT DE L'APPLICATION
# ===============================
if __name__ == "__main__":
    main()