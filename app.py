import streamlit as st
import pandas as pd
import os
import nltk
nltk.download('wordnet')


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
from nltk.stem import WordNetLemmatizer

# Load & preprocess movies
@st.cache_data
def load_movies(file_path="imdb_top_1000.csv"):
    movies = pd.read_csv(file_path)

    movies['Meta_score'] = movies['Meta_score'] / 10
    movies = movies[movies['Meta_score'].notnull()]

    for col in ['Director', 'Star1', 'Star2', 'Star3', 'Star4']:
        if col in movies.columns:
            movies[col] = movies[col].astype(str).str.replace(" ", "")

    movies['text_data'] = (
        movies['Series_Title'].astype(str) + ' ' +
        movies['Overview'].astype(str) + ' ' +
        movies['Director'].astype(str) + ' ' +
        movies['Star1'].astype(str) + ' ' +
        movies['Star2'].astype(str) + ' ' +
        movies['Star3'].astype(str) + ' ' +
        movies['Genre'].astype(str)
    )

    lemmatizer = WordNetLemmatizer()
    movies['text_data'] = movies['text_data'].apply(
        lambda x: ' '.join(lemmatizer.lemmatize(w) for w in x.split())
    )

    if 'My_rating ' not in movies.columns:
        movies['My_rating '] = None

    return movies


# Save / Load user ratings

def save_user_ratings(movies, path="my_ratings.csv"):
    movies[['Series_Title', 'My_rating ']].dropna().to_csv(path, index=False)

def load_user_ratings(movies, path="my_ratings.csv"):
    if not os.path.exists(path):
        return movies

    ratings = pd.read_csv(path)
    movies = movies.merge(ratings, on='Series_Title', how='left', suffixes=('', '_saved'))
    movies['My_rating '] = movies['My_rating '].fillna(movies['My_rating _saved'])
    movies.drop(columns=['My_rating _saved'], inplace=True)
    return movies


# Train & recommend

def train_and_recommend(movies, top_n=10):
    rated = movies[movies['My_rating '].notnull()]
    unrated = movies[movies['My_rating '].isnull()]

    tfidf = TfidfVectorizer(max_features=2000)

    X_rated = tfidf.fit_transform(rated['text_data'])
    X_unrated = tfidf.transform(unrated['text_data'])

    X_train = hstack([X_rated, csr_matrix(rated[['IMDB_Rating']])])
    y_train = rated['My_rating '].values

    X_unrated = hstack([X_unrated, csr_matrix(unrated[['IMDB_Rating']])])

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_unrated)
    unrated = unrated.copy()
    unrated['Predicted_My_Rating'] = preds

    return unrated.sort_values(
        by='Predicted_My_Rating',
        ascending=False
    ).head(top_n)


# Streamlit UI

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Personalized Movie Recommendation System")

movies = load_movies()
movies = load_user_ratings(movies)


# Rating section

st.subheader("⭐ Rate a Movie")

movie_name = st.selectbox(
    "Choose a movie",
    movies['Series_Title'].unique()
)

rating = st.slider(
    "Your Rating",
    min_value=1.0,
    max_value=10.0,
    step=0.1
)

if st.button("Save Rating"):
    movies.loc[movies['Series_Title'] == movie_name, 'My_rating '] = rating
    save_user_ratings(movies)
    st.success(f"Saved rating for **{movie_name}**")

# ----------------------------
# Recommendation section
# ----------------------------
st.subheader("🎯 Recommended Movies")

if st.button("Get Recommendations"):
    if movies['My_rating '].notnull().sum() < 2:
        st.warning("Please rate at least 2 movies first.")
    else:
        recs = train_and_recommend(movies)
        st.dataframe(
            recs[['Series_Title', 'Predicted_My_Rating']],
            use_container_width=True
        )
