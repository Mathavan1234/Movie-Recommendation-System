import streamlit as st
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

# Setting up the Web App Configuration
st.set_page_config(page_title="MovieRec", page_icon="🍿", layout="wide")
st.markdown(f"""
            <style>
            .stApp {{background-image: url("");
                     background-attachment: fixed;
                     base: light;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# Loading the models which were saved using JobLib
df = joblib.load('models/movie_db.df')
tfidf_matrix = joblib.load('models/tfidf_mat.tf')
tfidf = joblib.load('models/vectorizer.tf')
cos_mat = joblib.load('models/cos_mat.mt')


def get_recommendations(movie):
    # Extracting the Movie index from the DataFrame
    index = df[df['title'] == movie].index[0]

    # Sorting Top N Similar Movies
    similar_movies = sorted(
        list(enumerate(cos_mat[index])), reverse=True, key=lambda x: x[1])

    # Extracting similar movie names and returning it from the DataFrame
    recomm = []
    for i in similar_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
    return recomm


def get_keywords_recommendations(keywords):
    keywords = keywords.split()
    keywords = " ".join(keywords)

    # Transforming the Keyword String to Numerical Vector Format
    key_tfidf = tfidf.transform([keywords])

    # Computing the Cosine Similarity
    result = cosine_similarity(key_tfidf, tfidf_matrix)

    # Sorting Top N Simialar Movies
    similar_key_movies = sorted(
        list(enumerate(result[0])), reverse=True, key=lambda x: x[1])

    # Extracting similar movie names and returning it from the DataFrame
    recomm = []
    for i in similar_key_movies[1:6]:
        recomm.append(df.iloc[i[0]].title)
    return recomm


def get_genre_recommendations(movie_title):
    genres_list = [genre.split(",") for genre in df['All Genres']]

    # To help in binary matrix genre text representation
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(genres_list)

    # To initialize the KNN model which has 5 nearest neighbours for recommendation
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')

    # To train the KNN model on the genre matrix
    knn.fit(genre_matrix)

    # To check if the movie title exists in the Data Frame
    if movie_title not in df['title'].values:
        return f"Movie '{movie_title}' not found in the database."

    # To get the particular movie index searched
    index = df[df['title'] == movie_title].index[0]

    # To calculate the nearest neighbors based on genre using KNN algorithm
    indices = knn.kneighbors([genre_matrix[index]])

    # To extract the recommended movies based on genre
    recommended_movies = []
    for i in indices[1][0][1:6]:
        recommended_movies.append(df.iloc[i]['title'])

    return recommended_movies


def combined_recommendations(movie_title, keywords, n=5, keyword_weight=0.5):

    # To call and get recommendations from both methods
    keyword_rec = get_keywords_recommendations(keywords)
    genre_rec = get_genre_recommendations(movie_title)

    # To create a dictionary to store movie recommendations and their scores
    recommendations = {}

    # To assign scores to movies based on keyword-based recommendations
    for movie in keyword_rec:
        # It updates the weighted score of the keywords, if movie is not present default is 0
        recommendations[movie] = recommendations.get(movie, 0) + keyword_weight

    # To assign scores to movies based on genre-based recommendations
    for movie in genre_rec:
        # It updates the weighted score of the genres, if movie is not present default is 0
        # 1 - score to give less weight to movies recommended by keywords
        recommendations[movie] = recommendations.get(
            movie, 0) + (1 - keyword_weight)

    # To sort movies in descending order by their combined scores
    combined_rec = sorted(recommendations.items(),
                          key=lambda x: x[1], reverse=True)

    # To return the recommended movies to the user
    return [movie[0] for movie in combined_rec[:n]]

# Function to fetch Movie Posters from MovieDB API


def fetch_poster(movies):
    ids = []
    posters = []
    for i in movies:
        ids.append(df[df.title == i]['id'].values[0])

    for i in ids:
       # Constructing the URL for accessing the MovieDB API with the movie ID
        url = f"https://api.themoviedb.org/3/movie/{i}?api_key=43569c4e47535e6f378daf211b4ec4f9"

        # Making a GET request to the API endpoint and convert the response to JSON format
        data = requests.get(url)
        data = data.json()

        # Extracting the poster path from the API response
        poster_path = data['poster_path']

        # Constructing the full URL for the movie poster using the poster path
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

        # Appending the full poster URL to the posters empty list created before
        posters.append(full_path)

    # Returning the list of poster URLs as the output of the function
    return posters


# App Layout
st.image("images/appmainlogo.png")
st.title("Movie Finder 🤖")
posters = 0
movies = 0

with st.sidebar:
    st.image("images/appsidelogo.png", use_column_width=True)
    st.header("Get Recommendations by")
    search_type = st.radio(
        "", ('Movie Title', 'Keywords', 'Genres', 'Hybrid (Keyword + Genre) Filtering'))

# call functions based on selectbox
if search_type == 'Movie Title':
    st.subheader("Select Movie 🎬")
    movie_name = st.selectbox('', df.title)
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_recommendations(movie_name)
            posters = fetch_poster(movies)

elif search_type == 'Keywords':
    st.subheader('Enter Cast / Crew / Overview 🎞️')
    keyword = st.text_input('', 'Christopher Nolan')
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_keywords_recommendations(keyword)
            posters = fetch_poster(movies)

elif search_type == "Genres":
    st.subheader('Enter Movie of preferred Genre 🎦')
    genre = st.selectbox('', df.title)
    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = get_genre_recommendations(genre)
            posters = fetch_poster(movies)

else:
    st.subheader('Enter Cast / Crew / Overview 🎞️')
    keyword_hybrid = st.text_input('', 'Daniel Radcliffe')
    st.subheader('Enter Movie of preferred Genre 🎦')
    genre_hybrid = st.selectbox('', df.title)

    if st.button('Recommend 🚀'):
        with st.spinner('Wait for it...'):
            movies = combined_recommendations(genre_hybrid, keyword_hybrid)
            posters = fetch_poster(movies)

# Display Posters - this change was done becuse similar genres of the movies can be less than 5
if movies and posters:
    # Limited the number of movies to display to maximum of 5 (ATMOST 5 MOVIES TO DISPLAY)
    num_movies = min(5, len(movies))

    # Create columns to display the movies and posters
    cols = st.columns(num_movies, gap='medium')

    # Loop through the number of movies to display
    for i in range(num_movies):
        # For each column, display movie title and poster image
        with cols[i]:
            # Check if the index is within the length of both movies and posters list
            if i < len(movies) and i < len(posters):
                # Display the movie title and poster image in the current column
                st.text(movies[i])
                st.image(posters[i])
