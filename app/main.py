# main.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from pathlib import Path
import hashlib
import sqlite3

# st.title('Filmofil')

# Naloadamo podatke
@st.cache_data
def load_data():
    movies = pd.read_csv("./data/movies.csv")
    ratings = pd.read_csv("./data/ratings.csv")
    cast = pd.read_csv("./data/cast.csv")
    return movies, ratings, cast

movies, ratings, cast = load_data()

# Process movies to extract year
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '').str.strip()

# ----- User Authentication ----- #
def create_connection():
    conn = sqlite3.connect("users.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS user_ratings (
                        username TEXT,
                        movieId INTEGER,
                        rating REAL)''')
    return conn

conn = create_connection()

# Hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    return cur.fetchone() is not None

def logout_user():
    return "username" in st.session_state

# ----- App Navigation ----- #
#page = st.sidebar.radio("Izberi stran", ["Analiza podatkov", "Primerjava filmov", "Priporočila", "Prijava / Registracija"])
# Initialize page in session_state if not set
if "page" not in st.session_state:
    st.session_state.page = "Analiza podatkov"

st.sidebar.markdown("## Navigacija")
pages = ["Analiza podatkov", "Primerjava filmov", "Priporočila", "Prijava"]
for p in pages:
    if st.sidebar.button(p):
        st.session_state.page = p

# Get active page
page = st.session_state.page


# ----- ANALIZA PODATKOV ----- #
if page == "Analiza podatkov":
    st.title("Top 10 filmov po ocenah")

    min_ratings = st.slider("Minimalno število ocen", 1, 350, 10)
    genre = st.selectbox("Izberi žanr", ["Vsi"] + sorted(set('|'.join(movies['genres']).split('|'))))
    year = st.selectbox("Izberi leto", ["Vsa leta"] + sorted(movies['year'].dropna().astype(int).unique().tolist()))

    df = ratings.groupby("movieId").agg(avg_rating=("rating", "mean"),
                                          count_rating=("rating", "count")).reset_index()
    df = df.merge(movies, on="movieId")

    if genre != "Vsi":
        df = df[df['genres'].str.contains(genre)]
    if year != "Vsa leta":
        df = df[df['year'] == int(year)]

    df = df[df['count_rating'] >= min_ratings]
    df = df.sort_values("avg_rating", ascending=False).head(10)

    st.dataframe(df[["title", "avg_rating", "count_rating"]])

# ----- PRIMERJAVA FILMOV ----- #
elif page == "Primerjava filmov":
    st.title("Primerjava dveh filmov")
    movie_names = movies["title"].tolist()
    movie1 = st.selectbox("Izberi prvi film", movie_names)
    movie2 = st.selectbox("Izberi drugi film", movie_names, index=1)

    def movie_stats(title):
        mid = movies[movies.title == title].movieId.values[0]
        df = ratings[ratings.movieId == mid]
        df["year"] = pd.to_datetime(df["timestamp"], unit="s").dt.year
        return {
            "avg": df.rating.mean(),
            "std": df.rating.std(),
            "count": len(df),
            "hist": df,
            "yearly_avg": df.groupby("year")["rating"].mean().reset_index(),
            "yearly_count": df.groupby("year")["rating"].count().reset_index(name="count")
        }

    col1, col2 = st.columns(2)
    with col1:
        s1 = movie_stats(movie1)
        st.subheader(movie1)
        st.metric("Povprečna ocena", f"{s1['avg']:.2f}")
        st.metric("Število ocen", s1['count'])
        st.metric("Standardni odklon", f"{s1['std']:.2f}")
        st.altair_chart(alt.Chart(s1['hist']).mark_bar().encode(x="rating:Q", y='count():Q'), use_container_width=True)
        st.altair_chart(alt.Chart(s1['yearly_avg']).mark_line().encode(x='year:O', y='rating:Q'), use_container_width=True)
        st.altair_chart(alt.Chart(s1['yearly_count']).mark_bar().encode(x='year:O', y='count:Q'), use_container_width=True)

    with col2:
        s2 = movie_stats(movie2)
        st.subheader(movie2)
        st.metric("Povprečna ocena", f"{s2['avg']:.2f}")
        st.metric("Število ocen", s2['count'])
        st.metric("Standardni odklon", f"{s2['std']:.2f}")
        st.altair_chart(alt.Chart(s2['hist']).mark_bar().encode(x="rating:Q", y='count():Q'), use_container_width=True)
        st.altair_chart(alt.Chart(s2['yearly_avg']).mark_line().encode(x='year:O', y='rating:Q'), use_container_width=True)
        st.altair_chart(alt.Chart(s2['yearly_count']).mark_bar().encode(x='year:O', y='count:Q'), use_container_width=True)

# ----- PRIPOROČILA ----- #
elif page == "Priporočila":
    if "username" not in st.session_state:
        st.warning("Najprej se prijavi.")
    else:
        st.title("Priporočila za: " + st.session_state.username)

        cur = conn.cursor()
        cur.execute("SELECT movieId, rating FROM user_ratings WHERE username=?", (st.session_state.username,))
        user_data = pd.DataFrame(cur.fetchall(), columns=["movieId", "rating"])
        #user_data["movieId"] = user_data["movieId"].astype(int)
        # Decode byte strings and convert to int
        user_data["movieId"] = user_data["movieId"].apply(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else int(x))


        st.subheader("Ocenjeni filmi")
        rated = user_data.merge(movies, on="movieId")
        st.dataframe(rated[["title", "rating"]])

        unrated = movies[~movies.movieId.isin(user_data.movieId)]

        st.subheader("Oceni več filmov")
        movie_to_rate = st.selectbox("Izberi film za oceno", unrated.title.tolist())
        rating_val = st.slider("Ocena", 0.5, 5.0, 3.0, 0.5)
        if st.button("Shrani oceno"):
            mid = movies[movies.title == movie_to_rate].movieId.values[0]
            conn.execute("INSERT INTO user_ratings (username, movieId, rating) VALUES (?, ?, ?)",
                         (st.session_state.username, mid, rating_val))
            conn.commit()
            st.success("Ocena shranjena!")

        if len(user_data) >= 10:
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.feature_extraction.text import TfidfVectorizer

            user_vector = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
            my_vec = np.zeros(user_vector.shape[1])
            for i, m in enumerate(user_vector.columns):
                if m in user_data.movieId.values:
                    my_vec[i] = user_data[user_data.movieId == m].rating.values[0]

            similarity = cosine_similarity([my_vec], user_vector.values)[0]
            top_users = user_vector.index[np.argsort(similarity)[-5:]]
            recs = user_vector.loc[top_users].mean().sort_values(ascending=False)
            recs = recs.drop(user_data.movieId.values, errors='ignore')
            recs = recs.head(10).reset_index()
            recs = recs.merge(movies, on="movieId")
            st.subheader("Priporočeni filmi")
            st.dataframe(recs[["title"]])
        else:
            st.info("Za priporočila morate oceniti vsaj 10 filmov.")

# ----- REGISTRACIJA/PRIJAVA ----- #
elif page == "Prijava":
    st.title("Prijava")
    form = st.form("auth")
    choice = form.radio("Izberi", ["Prijava", "Registracija"])
    user = form.text_input("Uporabniško ime")
    pwd = form.text_input("Geslo", type="password")
    submit = form.form_submit_button("Potrdi")

    if submit:
        if choice == "Registracija":
            if register_user(user, pwd):
                st.success("Uporabnik registriran. Prijavi se.")
            else:
                st.error("Uporabniško ime že obstaja.")
        else:
            if login_user(user, pwd):
                st.success("Prijava uspešna.")
                st.session_state.username = user
            else:
                st.error("Napačni podatki.")

    if st.button("Odjava"):
        if logout_user():
            st.success("Odjava uspešna.")
            del st.session_state.username
        else:
            st.error("Nihče ni bil prijavljen.")
