import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
from collections import Counter, defaultdict
import pickle
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Configuration initiale
warnings.filterwarnings('ignore')
plt.style.use('default')

# ===== CONFIGURATION STREAMLIT =====
st.set_page_config(
    page_title="Analyse de Sentiment - Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== CONSTANTES =====
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off'
}

RANDOM_STATE = 42
MAX_VOCAB_SIZE = 2000  # Réduit pour les performances
MAX_SAMPLES = 5000     # Limite pour les gros datasets

# ===== FONCTIONS UTILITAIRES =====
def clean_text(text):
    """Nettoie le texte de manière robuste"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    # Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Garder seulement les lettres et espaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize_text(text, ngram_range=(1, 1)):
    """Tokenise le texte et crée des n-grammes"""
    if not text:
        return []
    
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    if not words:
        return []
    
    tokens = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(words) - n + 1):
            tokens.append(' '.join(words[i:i + n]))
    
    return tokens

def create_vocabulary(texts, max_features=MAX_VOCAB_SIZE):
    """Crée le vocabulaire à partir des textes"""
    word_freq = Counter()
    
    for text in texts:
        tokens = tokenize_text(text, ngram_range=(1, 2))
        word_freq.update(tokens)
    
    # Garder les mots les plus fréquents
    vocab = {word: idx for idx, (word, _) in enumerate(word_freq.most_common(max_features))}
    return vocab

def compute_tf_idf(texts, vocab):
    """Calcule les vecteurs TF-IDF"""
    n_docs = len(texts)
    n_features = len(vocab)
    
    # Calculer IDF
    df = Counter()
    for text in texts:
        tokens = set(tokenize_text(text, ngram_range=(1, 2)))
        for token in tokens:
            if token in vocab:
                df[token] += 1
    
    idf = {}
    for word in vocab:
        idf[word] = np.log(n_docs / (df[word] + 1)) + 1
    
    # Calculer TF-IDF pour chaque document
    tfidf_matrix = np.zeros((n_docs, n_features))
    
    for doc_idx, text in enumerate(texts):
        tokens = tokenize_text(text, ngram_range=(1, 2))
        tf = Counter(tokens)
        
        for token, count in tf.items():
            if token in vocab:
                word_idx = vocab[token]
                tfidf_matrix[doc_idx, word_idx] = count * idf[token]
        
        # Normalisation L2
        norm = np.linalg.norm(tfidf_matrix[doc_idx])
        if norm > 0:
            tfidf_matrix[doc_idx] /= norm
    
    return tfidf_matrix, vocab, idf

# ===== FONCTIONS MISES EN CACHE =====
@st.cache_data(show_spinner=False)
def load_data():
    """Charge les données avec gestion d'erreurs"""
    try:
        # Essayer différents noms de fichiers courants
        possible_files = [
            "IMDB Dataset.csv",
            "imdb_dataset.csv", 
            "IMDB_Dataset.csv",
            "dataset.csv"
        ]
        
        df = None
        loaded_file = None
        
        for filename in possible_files:
            try:
                df = pd.read_csv(filename, encoding='utf-8')
                loaded_file = filename
                break
            except (FileNotFoundError, UnicodeDecodeError):
                try:
                    df = pd.read_csv(filename, encoding='latin-1')
                    loaded_file = filename
                    break
                except:
                    continue
        
        if df is None:
            st.error(f"❌ Aucun fichier de données trouvé. Cherché: {', '.join(possible_files)}")
            st.info("📁 Assurez-vous qu'un fichier CSV avec les colonnes 'review' et 'sentiment' soit présent.")
            return None
            
        # Vérifier les colonnes
        required_cols = ['review', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Colonnes manquantes. Trouvé: {list(df.columns)}, Requis: {required_cols}")
            return None
        
        # Nettoyage initial
        df = df.dropna().drop_duplicates()
        df['review'] = df['review'].apply(clean_text)
        df = df[df['review'].str.len() > 10]  # Garder seulement les reviews avec du contenu
        
        # Limiter la taille pour les performances
        if len(df) > MAX_SAMPLES:
            df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE)
            st.info(f"📊 Dataset limité à {MAX_SAMPLES} échantillons pour optimiser les performances")
        
        st.success(f"✅ Données chargées: {len(df)} échantillons depuis '{loaded_file}'")
        return df
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def prepare_data(_df):
    """Prépare les données pour l'entraînement"""
    if _df is None or _df.empty:
        return None, None, None, None, None
    
    try:
        # Préparer les features TF-IDF
        texts = _df['review'].tolist()
        X, vocab, idf = compute_tf_idf(texts, create_vocabulary(texts))
        
        # Encoder les labels
        le = LabelEncoder()
        y = le.fit_transform(_df['sentiment'])
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, {'vocab': vocab, 'idf': idf, 'le': le}
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la préparation des données: {str(e)}")
        return None, None, None, None, None

# ===== FONCTIONS DE VISUALISATION =====
def create_confusion_matrix(y_true, y_pred, class_names):
    """Crée une matrice de confusion avec Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Matrice de Confusion",
        labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(
        width=500,
        height=400,
        title_x=0.5
    )
    
    return fig

def create_roc_curve(y_true, y_proba):
    """Crée une courbe ROC avec Plotly"""
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Ligne de base',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Courbe ROC',
        xaxis_title='Taux de Faux Positifs',
        yaxis_title='Taux de Vrais Positifs',
        width=500,
        height=400,
        title_x=0.5
    )
    
    return fig

def create_precision_recall_curve(y_true, y_proba):
    """Crée une courbe Precision-Recall avec Plotly"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.3f})',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Courbe Précision-Rappel',
        xaxis_title='Rappel',
        yaxis_title='Précision',
        width=500,
        height=400,
        title_x=0.5
    )
    
    return fig

def predict_sentiment(text, model, preprocessor):
    """Prédit le sentiment d'un texte"""
    try:
        if not text or not text.strip():
            return None, None
        
        # Nettoyer et vectoriser le texte
        clean = clean_text(text)
        if not clean:
            return None, None
        
        tokens = tokenize_text(clean, ngram_range=(1, 2))
        if not tokens:
            return None, None
        
        # Créer le vecteur TF-IDF
        vocab = preprocessor['vocab']
        idf = preprocessor['idf']
        
        tf = Counter(tokens)
        vector = np.zeros(len(vocab))
        
        for token, count in tf.items():
            if token in vocab:
                idx = vocab[token]
                vector[idx] = count * idf[token]
        
        # Normaliser
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        # Prédire
        prediction = model.predict(vector.reshape(1, -1))[0]
        probabilities = model.predict_proba(vector.reshape(1, -1))[0]
        
        sentiment = preprocessor['le'].inverse_transform([prediction])[0]
        
        return sentiment, probabilities
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None, None

# ===== INTERFACE PRINCIPALE =====
def main():
    # En-tête
    st.markdown('<h1 class="main-header">🎬 Analyse de Sentiment - Avis de Films</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">👨‍💻 Développé par Messanh Yaovi KODJO</p>', unsafe_allow_html=True)
    
    # Chargement des données
    with st.spinner("📂 Chargement des données..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar - Affichage des données
    if st.sidebar.checkbox("👀 Afficher les données"):
        st.subheader("📊 Aperçu des données")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Total", len(df))
        with col2:
            pos_count = (df['sentiment'] == 'positive').sum()
            st.metric("😊 Positifs", pos_count)
        with col3:
            neg_count = (df['sentiment'] == 'negative').sum()
            st.metric("😞 Négatifs", neg_count)
        
        # Échantillon des données
        sample_df = df.sample(min(10, len(df)), random_state=RANDOM_STATE)
        st.dataframe(sample_df[['review', 'sentiment']], use_container_width=True)
        
        # Graphique de distribution
        fig = px.pie(
            values=[pos_count, neg_count],
            names=['Positif', 'Négatif'],
            title="Distribution des Sentiments",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sidebar - Configuration du modèle
    st.sidebar.header("⚙️ Configuration")
    
    model_type = st.sidebar.selectbox(
        "🤖 Choix du modèle",
        ["Logistic Regression", "Random Forest"],
        help="Sélectionnez l'algorithme de classification"
    )
    
    # Hyperparamètres
    if model_type == "Logistic Regression":
        max_iter = st.sidebar.slider("🔄 Itérations max", 100, 1000, 300, 50)
        C = st.sidebar.slider("📊 Régularisation (C)", 0.1, 10.0, 1.0, 0.1)
    else:
        n_estimators = st.sidebar.slider("🌳 Nombre d'arbres", 10, 100, 50, 10)
        max_depth = st.sidebar.selectbox("📏 Profondeur max", [None, 5, 10, 15, 20])
    
    # Métriques à afficher
    show_metrics = st.sidebar.multiselect(
        "📈 Métriques à afficher",
        ["Matrice de confusion", "Courbe ROC", "Courbe Précision-Rappel"],
        default=["Matrice de confusion"]
    )
    
    # Bouton d'entraînement
    if st.sidebar.button("🚀 Entraîner le modèle", type="primary"):
        
        # Préparation des données
        with st.spinner("🔧 Préparation des données..."):
            data_result = prepare_data(df)
        
        if data_result[0] is None:
            st.error("❌ Erreur lors de la préparation des données")
            st.stop()
        
        X_train, X_test, y_train, y_test, preprocessor = data_result
        
        # Configuration du modèle
        with st.spinner("🏋️ Entraînement en cours..."):
            try:
                if model_type == "Logistic Regression":
                    model = LogisticRegression(
                        max_iter=max_iter,
                        C=C,
                        random_state=RANDOM_STATE,
                        solver='liblinear'
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Métriques
                accuracy = accuracy_score(y_test, y_pred)
                
                # Sauvegarder dans session state
                st.session_state['model'] = model
                st.session_state['preprocessor'] = preprocessor
                st.session_state['accuracy'] = accuracy
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['y_proba'] = y_proba
                
                st.success(f"✅ Modèle entraîné avec succès! Précision: {accuracy:.3f}")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
                st.stop()
    
    # Affichage des résultats
    if 'model' in st.session_state:
        st.header("📊 Résultats du Modèle")
        
        # Métriques principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Précision", f"{st.session_state['accuracy']:.3f}")
        with col2:
            st.metric("📊 Échantillons test", len(st.session_state['y_test']))
        with col3:
            st.metric("🤖 Modèle", model_type)
        
        # Rapport détaillé
        with st.expander("📋 Rapport de classification détaillé"):
            class_names = st.session_state['preprocessor']['le'].classes_
            report = classification_report(
                st.session_state['y_test'], 
                st.session_state['y_pred'], 
                target_names=class_names
            )
            st.code(report)
        
        # Visualisations
        if show_metrics:
            st.subheader("📈 Visualisations")
            
            cols = st.columns(len(show_metrics))
            class_names = st.session_state['preprocessor']['le'].classes_
            
            for idx, metric in enumerate(show_metrics):
                with cols[idx]:
                    if metric == "Matrice de confusion":
                        fig = create_confusion_matrix(
                            st.session_state['y_test'],
                            st.session_state['y_pred'],
                            class_names
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif metric == "Courbe ROC":
                        fig = create_roc_curve(
                            st.session_state['y_test'],
                            st.session_state['y_proba']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif metric == "Courbe Précision-Rappel":
                        fig = create_precision_recall_curve(
                            st.session_state['y_test'],
                            st.session_state['y_proba']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Section de prédiction
    st.header("🔮 Prédiction de Sentiment")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle!")
        st.info("👆 Utilisez la configuration dans la barre latérale")
    else:
        # Interface de prédiction
        user_text = st.text_area(
            "✍️ Entrez votre avis de film:",
            placeholder="Ex: This movie was absolutely fantastic! Great acting and amazing plot...",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            predict_btn = st.button("🔮 Analyser", type="primary")
        
        if predict_btn and user_text.strip():
            with st.spinner("🤔 Analyse en cours..."):
                sentiment, probabilities = predict_sentiment(
                    user_text,
                    st.session_state['model'],
                    st.session_state['preprocessor']
                )
            
            if sentiment:
                # Affichage du résultat
                if sentiment.lower() == 'positive':
                    st.success(f"😊 **Sentiment: {sentiment.upper()}**")
                    st.balloons()
                else:
                    st.error(f"😞 **Sentiment: {sentiment.upper()}**")
                
                # Probabilités
                if probabilities is not None:
                    col1, col2 = st.columns(2)
                    class_names = st.session_state['preprocessor']['le'].classes_
                    
                    for i, (prob, class_name) in enumerate(zip(probabilities, class_names)):
                        with col1 if i == 0 else col2:
                            emoji = "😞" if class_name == 'negative' else "😊"
                            st.metric(f"{emoji} {class_name.title()}", f"{prob:.3f}")
                    
                    # Graphique des probabilités
                    fig = go.Figure(data=[
                        go.Bar(
                            x=class_names,
                            y=probabilities,
                            marker_color=['#e74c3c' if name == 'negative' else '#2ecc71' for name in class_names]
                        )
                    ])
                    fig.update_layout(
                        title="Probabilités de Classification",
                        xaxis_title="Sentiment",
                        yaxis_title="Probabilité",
                        height=300,
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Impossible d'analyser ce texte. Essayez avec un texte plus long.")
        
        elif predict_btn:
            st.warning("⚠️ Veuillez entrer un texte à analyser!")
    
    # Section d'aide
    with st.expander("ℹ️ Guide d'utilisation"):
        st.markdown("""
        ### 🚀 Comment utiliser cette application:
        
        1. **📊 Explorer les données** (optionnel):
           - Cochez "Afficher les données" dans la barre latérale
           - Consultez les statistiques et la distribution
        
        2. **⚙️ Configurer le modèle**:
           - Choisissez l'algorithme (Logistic Regression ou Random Forest)
           - Ajustez les hyperparamètres selon vos besoins
           - Sélectionnez les métriques à visualiser
        
        3. **🚀 Entraîner**:
           - Cliquez sur "Entraîner le modèle"
           - Attendez la fin de l'entraînement
           - Consultez les résultats et visualisations
        
        4. **🔮 Prédire**:
           - Entrez votre propre avis de film
           - Cliquez sur "Analyser"
           - Obtenez le sentiment prédit avec les probabilités
        
        ### 🔧 Fonctionnalités techniques:
        - **Preprocessing**: Nettoyage automatique du texte
        - **Vectorisation**: TF-IDF avec unigrammes et bigrammes  
        - **Modèles**: Logistic Regression et Random Forest
        - **Métriques**: Accuracy, Précision, Rappel, F1-Score
        - **Visualisations**: Matrice de confusion, Courbes ROC et PR
        
        ### ⚡ Optimisations:
        - Mise en cache pour des performances optimales
        - Interface responsive avec Plotly
        - Gestion robuste des erreurs
        """)

if __name__ == '__main__':
    main()