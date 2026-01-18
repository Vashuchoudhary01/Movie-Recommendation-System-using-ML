import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Movie Recommendation System Using Machine Learning",
    page_icon="🎬",
    layout="wide"
)

# -------------------- UI CSS --------------------
st.markdown(
    """
    <style>
    .movie-card {
        padding:18px;
        border-radius:14px;
        background: linear-gradient(145deg, #1c1f26, #232733);
        text-align:center;
        height:200px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .movie-title {
        font-size:16px;
        font-weight:600;
        line-height:1.3;
        color:#f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["Tags"]).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie_title):
    if movie_title not in movies["title"].values:
        return [], []

    idx = movies[movies["title"] == movie_title].index[0]
    scores = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    titles, ratings = [], []
    for i, _ in scores:
        titles.append(movies.iloc[i].title)
        r = movies.iloc[i].vote_average
        ratings.append(5 if r >= 5 else 4 if r >= 4 else 2.5)

    return titles, ratings

# -------------------- SIDEBAR --------------------
st.sidebar.title("🎥 About Project")
st.sidebar.info(
    """
    **Movie Recommendation System**

    🔹 Content-Based Filtering  
    🔹 Cosine Similarity  
    🔹 Streamlit Frontend  
    🔹 Python + ML  

    Built for **Placement Showcase**
    """
)

# -------------------- MAIN UI --------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#f5c518;'>🎬 Movie Recommendation System</h1>
    <p style='text-align:center;color:#9aa0a6;'>
    Type a movie name and press Enter or click Recommend
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- INPUT (SEARCH + ENTER SUPPORT) --------------------
selected_movie = st.selectbox(
    "🎞️ Search or select a movie",
    movies["title"].values,
    index=None,
    placeholder="Type movie name..."
)

recommend_btn = st.button("✨ Recommend Movies")

# -------------------- TRIGGER LOGIC --------------------
if selected_movie and (recommend_btn or st.session_state.get("last_movie") != selected_movie):
    st.session_state["last_movie"] = selected_movie

    titles, ratings = recommend(selected_movie)

    if titles:
        st.success("✅ Movies you may like:")
        cols = st.columns(5)

        for i in range(5):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">🎬 {titles[i]}</div>
                        <div style="color:#f5c518;">Rating: {ratings[i]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.error("❌ Movie not found.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9aa0a6;'>🚀 Built with Streamlit | ML Placement Project</p>",
    unsafe_allow_html=True
)
