# 🎬 Movie Recommender

A **personalized movie recommendation app** built using Python and Streamlit. This project suggests movies based on IMDb data and your own movie ratings — helping you discover films you'll love.

🔗 **Live website:**  
https://keshav9926-movie-recommender-app-ofv4l1.streamlit.app/

---

## 💡 Overview

The Movie Recommender predicts movies you might enjoy based on your preferences and past ratings. It leverages the IMDb Top 1000 dataset and your input to surface tailored movie suggestions.

---

## 🔗 Try It Live

👉 **Open the app:**  
https://keshav9926-movie-recommender-app-ofv4l1.streamlit.app/

Share this link with friends to let them explore personalized movie recommendations!

---

## 🚀 Features

- 📊 Recommendations based on IMDb’s Top 1000 movies  
- ⭐ Personalized suggestions based on user ratings  
- 🎛 Interactive UI built with Streamlit  
- 🐍 Developed using Python and Pandas  

---

## 🗂️ Project Structure

```text
movie_recommender/
├── app.py                   # Main Streamlit application
├── imdb_top_1000.csv        # Movie dataset
├── my_ratings.csv           # Personal ratings
├── web.ipynb                # Notebook for model development
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
---

## 🚀 Installation (Run Locally)
1️⃣ Clone the repository
```
git clone https://github.com/keshav9926/movie_recommender.git
cd movie_recommender
```

2️⃣ Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

3️⃣ Install dependencies
```
pip install -r requirements.txt
```

4️⃣ Run the Streamlit app
```
streamlit run app.py
```
