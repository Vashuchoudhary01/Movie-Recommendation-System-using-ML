📌 **Project Overview**

This system recommends movies by analyzing movie metadata such as genres, keywords, overview, and cast.
It uses Natural Language Processing (NLP) techniques and Cosine Similarity to find movies that are most similar to the selected one.

🧠 **How It Works**
Movie metadata is combined into a single Tags column

Text data is converted into numerical vectors using CountVectorizer

Cosine Similarity is calculated between movie vectors

Based on similarity scores, the top 5 most similar movies are recommended.

🛠️ **Tech Stack**

Programming Language: Python

Frontend: Streamlit

Machine Learning:

CountVectorizer

Cosine Similarity

Libraries:

Pandas

NumPy

Scikit-learn

Model Storage: Pickle (.pkl)

Version Control: Git & GitHub

Deployment: Streamlit Community Cloud

📂 **Project Structure**
Movie-Recommendation-System/
│
├── app.py                  # Streamlit application
├── movie_dict.pkl          # Processed movie dataset
├── movie.pkl / similarity.pkl  # Similarity matrix
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── .gitattributes          # Git LFS configuration

✨ **Features**

🔍 Search movie by name

🎯 Content-based recommendations

⚡ Fast response using cached data

🎨 Clean & modern UI

🆓 Fully free deployment

👨‍💻 **Author**

Vashu Choudhary
📌 Aspiring Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/Vashuchoudhary01

LinkedIn: (https://www.linkedin.com/in/vashu-choudhary-47612a331/)
