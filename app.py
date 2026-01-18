import streamlit as st
import pickle
import pandas as pd

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Movie Recommender Using Machine Learning",
    page_icon="🎬",
    layout="wide"
)

# -------------------- SAFE UI CSS (NO THEME BREAKING) --------------------
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
        word-wrap:break-word;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOAD DATA --------------------
@st.cache_resource
def load_data():
    with open("movie_dict.pkl", "rb") as f:
        movies_dict = pickle.load(f)

    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    movies = pd.DataFrame(movies_dict)
    return movies, similarity

movies, similarity = load_data()

# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie_title):
    if movie_title not in movies["title"].values:
        return [], [], []

    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]
    titles = [movies.iloc[i[0]].title for i in movies_list]

    raw_ratings = [movies.iloc[i[0]].vote_average for i in movies_list]

    ratings = []
    for r in raw_ratings:
        if r >= 5:
            ratings.append(5)
        elif 4 <= r < 5:
            ratings.append(4)
        else:
            ratings.append(2.5)

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
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 **Created by:** Vashu Choudhary")

# -------------------- MAIN UI --------------------
st.markdown(
    """
    <h1 style='text-align:center;color:#f5c518;'>🎬 Movie Recommender System</h1>
    <p style='text-align:center;color:#9aa0a6;'>
    Select a movie to get recommendations
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns([3, 1])

with col1:
    selected_movie = st.selectbox(
        "🎞️ Select a movie",
        movies["title"].values
    )

with col2:
    st.write("")
    st.write("")
    recommend_btn = st.button("✨ Recommend Movies")

# -------------------- OUTPUT SECTION --------------------
if recommend_btn:
    with st.spinner("Finding best recommendations..."):
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
        st.error("❌ Movie not found in database.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9aa0a6;'>🚀 Built with Streamlit | ML Placement Project</p>",
    unsafe_allow_html=True
)
