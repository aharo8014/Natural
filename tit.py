# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import nltk
import networkx as nx
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from textblob import TextBlob
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models as gensimvis
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
import pyLDAvis
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import linregress

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos necesarios de forma robusta
required_resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'vader_lexicon': 'sentiment/vader_lexicon'
}

for resource, path in required_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Análisis Bibliométrico",
    layout="wide",  # Esto es lo más importante
    initial_sidebar_state="expanded"
)

# ----------- SESIÓN Y LOGIN ---------------- #
def login():
    st.title("🔐 Acceso a la Plataforma de Análisis Bibliométrico")
    user = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Ingresar"):
        if user == "admin" and password == "admin":
            st.session_state["auth"] = True
        else:
            st.error("Usuario o contraseña incorrectos")

def sidebar_menu():
    menu = [
        "🧩 1. Carga y Exploración Inicial",
        "📊 2. Bibliometría y Redes",
        "🧠 3. PLN y Minería de Texto",
        "❤️ 4. Sentimiento y Emoción",
        "🤖 5. Generación y Similitud",
        "🧭 6. Predicción y Recomendación"
    ]
    return st.sidebar.radio("Navegación", menu)

@st.cache_data
def cargar_datos(file):
    return pd.read_csv(file)

def filtrar_dataframe(df, columnas):
    return df[columnas] if columnas else df

# Estado global del archivo
if "df" not in st.session_state:
    st.session_state.df = None

# -------- SECCIÓN 1 -------- #
def seccion_1():
    st.header("🧩 SECCIÓN 1: CARGA Y EXPLORACIÓN INICIAL")

    uploaded_file = st.file_uploader("📤 Sube tu archivo CSV exportado de Scopus", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
            return
    else:
        st.info("🔄 Esperando archivo... Sube un archivo para continuar.")
        return

        # 2. Estadísticas básicas
        st.subheader("📈 Estadísticas Exploratorias")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Distribución por Año")
            if 'Year' in df.columns:
                year_count = df['Year'].value_counts().sort_index()
                fig = px.bar(x=year_count.index, y=year_count.values, labels={'x': 'Año', 'y': 'Cantidad'}, title="Publicaciones por Año")
                st.plotly_chart(fig)
        with col2:
            st.write("Distribución por Tipo de Documento")
            if 'Document Type' in df.columns:
                doc_type = df['Document Type'].value_counts()
                fig2 = px.pie(values=doc_type.values, names=doc_type.index, title="Tipo de Documento")
                st.plotly_chart(fig2)

        # 3. Filtros dinámicos
        st.subheader("🔍 Explorador Dinámico")
        columnas = st.multiselect("Selecciona columnas para ver", df.columns.tolist(), default=["Title", "Authors", "Year"])
        filtro = st.text_input("Buscar texto en títulos o autores")
        if filtro:
            df_filtrado = df[df['Title'].str.contains(filtro, case=False, na=False) | df['Authors'].str.contains(filtro, case=False, na=False)]
        else:
            df_filtrado = df
        st.dataframe(filtrar_dataframe(df_filtrado, columnas).head(50))

        # 4. Exportar
        st.subheader("⬇️ Exportar Resultados")
        csv = df_filtrado.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="resultados_filtrados.csv">📥 Descargar CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# -------- SECCIÓN 2 -------- #
def seccion_2():
    st.header("📊 SECCIÓN 2: BIBLIOMETRÍA Y REDES")
    df = st.session_state.df
    if df is None:
        st.warning("Por favor, sube un archivo en la Sección 1.")
        return

    st.subheader("📌 Productividad por Autor")
    if 'Authors' in df.columns:
        authors_series = df['Authors'].dropna().str.split(';').explode().str.strip()
        top_authors = authors_series.value_counts().head(20)
        fig = px.bar(x=top_authors.values, y=top_authors.index, orientation='h', title="Top 20 Autores Más Productivos", labels={"x": "Publicaciones"})
        st.plotly_chart(fig)

    st.subheader("📚 Productividad por Revista")
    if 'Source title' in df.columns:
        top_sources = df['Source title'].value_counts().head(20)
        fig = px.bar(x=top_sources.values, y=top_sources.index, orientation='h', title="Top 20 Revistas", labels={"x": "Publicaciones"})
        st.plotly_chart(fig)

    st.subheader("🌍 Países e Instituciones")
    if 'Affiliations' in df.columns:
        affiliations = df['Affiliations'].dropna().str.split(';').explode().str.strip()
        top_affiliations = affiliations.value_counts().head(15)
        fig = px.bar(x=top_affiliations.values, y=top_affiliations.index, orientation='h',
                     title="Afiliaciones más frecuentes")
        st.plotly_chart(fig)

    st.subheader("🤝 Red de Coautorías")
    coauthors = df['Authors'].dropna().str.split(';')
    G = nx.Graph()
    for author_list in coauthors:
        authors = [a.strip() for a in author_list]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                G.add_edge(authors[i], authors[j])

    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:20]
    subgraph = G.subgraph(dict(top_nodes).keys())
    pos = nx.spring_layout(subgraph, seed=42)

    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} ({G.degree[node]})")

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none'),
            go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                       marker=dict(size=10, color='blue'))
        ],
        layout=go.Layout(
            title='Red de Coautorías (Top 20)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# -------- SECCIÓN 3 -------- #
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE

def entrenar_nmf(textos, n_topics=5, max_features=5000):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                 max_features=max_features,
                                 stop_words='english')
    X = vectorizer.fit_transform(textos)
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)   # documentos × tópicos
    H = nmf.components_        # tópicos × términos
    vocab = np.array(vectorizer.get_feature_names_out())
    return nmf, W, H, vocab


def seccion_3():
    st.header("🧠 SECCIÓN 3: PLN y Minería de Texto")
    df = st.session_state.df
    if df is None or 'Abstract' not in df.columns:
        st.warning("No hay columna 'Abstract' en los datos.")
        return

    # ---------- Preprocesamiento ---------------- #
    st.subheader("🧹 Limpieza y Preprocesamiento")
    if 'Processed_Text' not in df.columns:
        df['Processed_Text'] = df['Abstract'].fillna("").apply(preprocess_text)
    st.dataframe(df[['Title', 'Processed_Text']].head())

    # ---------- Parámetros del modelo ----------- #
    st.subheader("⚙️ Parámetros del modelo de tópicos")
    col_a, col_b = st.columns(2)
    with col_a:
        k_topics = st.slider("Número de tópicos (k)", 2, 15, 5)
    with col_b:
        top_n = st.slider("Palabras por tópico", 5, 20, 10)

    # ---------- Entrenamiento ------------------- #
    with st.spinner("Entrenando NMF…"):
        nmf, W, H, vocab = entrenar_nmf(df['Processed_Text'],
                                        n_topics=k_topics)
    st.success("Modelo entrenado ✔️")

    # ---------- Visualización de términos ------- #
    st.subheader("📊 Impacto de los términos en cada tópico")
    topic_options = {f"Tópico {i+1}": i for i in range(k_topics)}
    selected = st.selectbox("Selecciona tópico", list(topic_options.keys()))
    idx = topic_options[selected]
    top_terms = H[idx].argsort()[::-1][:top_n]
    term_weights = H[idx][top_terms]
    term_words = vocab[top_terms]

    fig_terms = px.bar(x=term_weights[::-1],
                       y=term_words[::-1],
                       orientation='h',
                       labels={'x': 'Peso (importancia)', 'y': 'Término'},
                       title=f"{selected} — Top {top_n} palabras")
    st.plotly_chart(fig_terms, use_container_width=True)

    # ---------- Distribución de documentos ------ #
    st.subheader("🌐 Distribución de documentos (t‑SNE)")
    if W.shape[0] > 50:       # reducir tiempo
        sample_idx = np.random.choice(W.shape[0], 50, replace=False)
        W_vis = W[sample_idx]
        titles_vis = df.iloc[sample_idx]['Title']
    else:
        W_vis = W
        titles_vis = df['Title']

    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    coords = tsne.fit_transform(W_vis)

    dominant_topic = W_vis.argmax(axis=1)
    fig_tsne = px.scatter(x=coords[:,0], y=coords[:,1],
                          color=dominant_topic.astype(str),
                          hover_name=titles_vis,
                          labels={'color': 'Tópico'},
                          title="Mapa 2D de documentos según tópicos")
    st.plotly_chart(fig_tsne, use_container_width=True)

# -------- SECCIÓN 4 -------- #

def seccion_4():
    st.header("❤️ SECCIÓN 4: Sentimiento y Polaridad")
    df = st.session_state.df
    if df is None or 'Abstract' not in df.columns:
        st.warning("Se necesita una columna 'Abstract'.")
        return

    # ---------- (Re)cálculo de columnas necesarias ---------- #
    if 'Sentiment' not in df.columns or 'PolarityVB' not in df.columns or 'Subjetividad' not in df.columns:
        sia = SentimentIntensityAnalyzer()
        df['Sentiment'] = df['Abstract'].fillna("").apply(lambda x: sia.polarity_scores(x)['compound'])
        df['PolarityVB'] = df['Sentiment'].apply(lambda x: "Positivo" if x > 0.05 else ("Negativo" if x < -0.05 else "Neutral"))
        df['Subjetividad'] = df['Abstract'].fillna("").apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # ---------- 2. Distribución de polaridad ------ #
    st.subheader("📊 Distribución de polaridad (histograma + KDE)")
    fig_hist = px.histogram(df, x='Sentiment', nbins=40, marginal='violin',
                            title="Valores de polaridad (‑1 a 1)")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---------- 3. Mapa de calor: año × polaridad-- #
    st.subheader("🔥 Mapa de calor ‑ Promedio de polaridad por año y tipo")
    if 'Year' in df.columns and 'Document Type' in df.columns:
        pivot = (df.pivot_table(index='Year',
                                columns='PolarityVB',
                                values='Sentiment',
                                aggfunc='mean')
                   .reindex(columns=['Negativo','Neutral','Positivo']))

        fig_heat = px.imshow(pivot,
                             labels=dict(color="Polaridad media"),
                             aspect="auto",
                             color_continuous_scale="RdYlGn",
                             title="Heatmap Año × Polaridad media")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---------- 4. Matriz scatter 3‑D ------------- #
    st.subheader("🧭 Exploración 3‑D: Sentiment × Subjetividad × Año")
    if 'Year' in df.columns:
        fig_3d = px.scatter_3d(df, x='Sentiment', y='Subjetividad', z='Year',
                               color='PolarityVB',
                               hover_data=['Title','Authors'],
                               title="Mapa 3‑D de artículos")
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---------- 5. Boxplots comparativos ---------- #
    st.subheader("📦 Boxplot de polaridad por tipo de documento")
    if 'Document Type' in df.columns:
        top_tipos = df['Document Type'].value_counts().head(6).index  # 6 principales
        fig_box = px.box(df[df['Document Type'].isin(top_tipos)],
                         x='Document Type', y='Sentiment', color='Document Type',
                         title="Distribución de polaridad por tipo de documento")
        st.plotly_chart(fig_box, use_container_width=True)

    # ---------- 6. Indicadores clave -------------- #
    st.subheader("🔢 Indicadores rápidos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Polaridad media", f"{df['Sentiment'].mean():.2f}")
    col2.metric("Subjetividad media", f"{df['Subjetividad'].mean():.2f}")
    col3.metric("Artículos analizados", df.shape[0])


# -------- SECCIÓN 5 -------- #

def seccion_5():
    st.header("🤖 SECCIÓN 5: Generación y Similitud")

    df = st.session_state.df
    if df is None or 'Processed_Text' not in df.columns:
        st.warning("Necesita texto procesado desde la Sección 3.")
        return

    # ---------- 1. Similaridad global ------------- #
    st.subheader("📏 Matriz de similitud completa (TF‑IDF + coseno)")
    tfidf = TfidfVectorizer(max_features=8000)
    tfidf_matrix = tfidf.fit_transform(df['Processed_Text'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    avg_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
    st.metric("🔁 Similitud promedio", f"{avg_sim:.3f}")
    st.dataframe(pd.DataFrame(sim_matrix).iloc[:10, :10])  # preview

    # ---------- 2. WordCloud fondo blanco --------- #
    st.subheader("☁️ Nube de Palabras (fondo blanco)")
    text_all = " ".join(df['Processed_Text'].values)
    wordcloud = WordCloud(width=900, height=400,
                          background_color='white',
                          max_words=250).generate(text_all)
    st.image(wordcloud.to_array(), use_column_width=True)

    # ---------- 3. Clustering + PCA 2‑D ----------- #
    st.subheader("📌 Clustering (KMeans) y proyección 2‑D")
    n_clusters = st.slider("Número de clústeres", 2, 12, 6)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(tfidf_matrix)
    df['Cluster'] = kmeans.labels_
    coords = PCA(n_components=2).fit_transform(tfidf_matrix.toarray())
    df['X'], df['Y'] = coords[:, 0], coords[:, 1]
    fig_scatter = px.scatter(df, x='X', y='Y', color='Cluster',
                             hover_data=['Title'],
                             title="Mapa 2‑D de documentos (PCA)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------- 4. LDA con gensim + pyLDAvis ------ #
    st.subheader("📊 Visualizador interactivo de tópicos (gensim + pyLDAvis)")
    num_topics = st.slider("Tópicos LDA", 3, 15, 5)

    # preparar corpus gensim
    token_lists = df['Processed_Text'].str.split().tolist()
    dictionary = Dictionary(token_lists)
    corpus = [dictionary.doc2bow(text) for text in token_lists]

    with st.spinner("Entrenando LDA y generando pyLDAvis…"):
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics,
                             random_state=42,
                             passes=10,
                             chunksize=1000,
                             update_every=1)
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')
        html = pyLDAvis.prepared_data_to_html(vis_data)
    components.html(html, height=600, width=1000, scrolling=True)


# -------- SECCIÓN 6 -------- #

def seccion_6():
    st.header("🚀 SECCIÓN 6: Detección de Términos Emergentes")

    df = st.session_state.df
    if df is None or 'Processed_Text' not in df.columns or 'Year' not in df.columns:
        st.warning("Se requiere texto procesado (Sección 3) y la columna 'Year'.")
        return

    # ---------- 1. Construir matriz TF‑IDF por año ---------------- #
    st.subheader("🔧 Preparando matriz TF‑IDF por año…")
    years = sorted(df['Year'].dropna().unique())
    docs_per_year = df.groupby('Year')['Processed_Text'].apply(lambda x: " ".join(x)).reindex(years, fill_value="")

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_year = vectorizer.fit_transform(docs_per_year)
    terms = vectorizer.get_feature_names_out()

    # Convertir a DataFrame: filas = términos, columnas = años
    tfidf_df = pd.DataFrame(tfidf_year.T.toarray(), index=terms, columns=years)

    # ---------- 2. Calcular tendencia (pendiente) ----------------- #
    st.subheader("📈 Tendencia de cada término")
    slopes = {}
    for term in terms:
        y = tfidf_df.loc[term].values
        if (y > 0).sum() < 3:        # necesita al menos 3 años con presencia
            continue
        # regresión lineal simple año vs peso TF‑IDF
        slope, _, _, p_value, _ = linregress(range(len(years)), y)
        slopes[term] = (slope, p_value)

    # DataFrame ordenado por pendiente descendente
    emergent_df = (pd.DataFrame(slopes, index=['slope', 'p'])
                     .T.sort_values('slope', ascending=False))

    # ---------- 3. Selección interactiva de términos emergentes --- #
    top_k = st.slider("👑 Mostrar top K términos emergentes", 5, 50, 10)
    emergent_top = emergent_df.head(top_k)

    st.dataframe(emergent_top.style.format({'slope':'{:.4f}', 'p':'{:.3f}'}))

    # ---------- 4. Visualización de la evolución ------------------ #
    st.subheader("📊 Evolución temporal de los términos seleccionados")
    sel_terms = st.multiselect("Elige términos a graficar",
                               emergent_top.index.tolist(),
                               default=emergent_top.index[:5].tolist())

    if sel_terms:
        fig = px.line(tfidf_df.loc[sel_terms].T,
                      x=tfidf_df.columns,
                      y=sel_terms,
                      labels={'value':'Peso TF‑IDF', 'x':'Año', 'variable':'Término'},
                      title="Curvas de crecimiento TF‑IDF de términos emergentes")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- 5. Nube de palabras de emergentes recientes ------- #
    st.subheader("☁️ Nube de palabras (sólo emergentes últimos 3 años)")
    recent_years = years[-3:]
    recent_text  = " ".join([docs_per_year[y] for y in recent_years])
    recent_wc    = WordCloud(width=800, height=400,
                             background_color="white",
                             colormap="viridis",
                             max_words=200).generate(recent_text)
    st.image(recent_wc.to_array(), use_column_width=True)

    # ---------- 6. Recomendación rápida --------------------------- #
    st.subheader("💡 Recomendación rápida de investigación")
    if not emergent_top.empty:
        suggestion = emergent_top.index[0]
        st.write(f"**Sugerencia:** profundizar en el término emergente "
                 f"**“{suggestion}”** (pendiente ≈ {emergent_top.iloc[0]['slope']:.4f}).")


def main():
    if "auth" not in st.session_state or not st.session_state["auth"]:
        login()
    else:
        st.sidebar.success("Sesión iniciada como admin")
        selected = sidebar_menu()

        if selected == "🧩 1. Carga y Exploración Inicial":
            seccion_1()
        elif selected == "📊 2. Bibliometría y Redes":
            seccion_2()
        elif selected == "🧠 3. PLN y Minería de Texto":
            seccion_3()
        elif selected == "❤️ 4. Sentimiento y Emoción":
            seccion_4()
        elif selected == "🤖 5. Generación y Similitud":
            seccion_5()
        elif selected == "🧭 6. Predicción y Recomendación":
            seccion_6()

if __name__ == "__main__":
    main()