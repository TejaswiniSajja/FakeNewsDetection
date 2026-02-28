📰**Fake News Detection using Machine Learning**

A machine learning–based web application that classifies news articles as FAKE or REAL using Natural Language Processing (NLP). The project includes model training, evaluation, and a user-friendly Streamlit interface for live predictions.


🚀 **Features**

Text classification using TF-IDF Vectorization

Machine Learning model trained on real-world news datasets

High accuracy (~98–99%)

Interactive Streamlit Web App

Clean, modular, and reproducible codebase


🛠 **Tech Stack**

Python

Scikit-learn

Pandas, NumPy

NLTK

Streamlit

Pickle


📂 **Project Structure**

FakeNewsDetection/
│

├── app.py                  # Streamlit web application

├── train_model.py          # Model training & saving

├── ck_columns.py           # Dataset column verification

├── Fake.csv                # Fake news dataset

├── True.csv                # Real news dataset

├── vectorizer.pkl          # Generated TF-IDF vectorizer (ignored in Git)

├── fakenewsmodel.pkl       # Trained ML model (ignored in Git)

├── requirements.txt        # Required dependencies

├── README.md               # Project documentation

├── .gitignore              # Ignored files & folders

└── venv/                   # Virtual environment (ignored)


📊 **Model Performance**

Accuracy: ~98–99%

Precision: High for both FAKE and TRUE classes

Recall: High reliability across classes


**Sample Output:**

Accuracy: 0.98+
FAKE  -> Precision: 0.99 | Recall: 0.98
TRUE  -> Precision: 0.98 | Recall: 0.99


⚙️ **How to Run the Project**

1️⃣ Clone the Repository
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate    # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model
python train_model.py

✔ This generates:

vectorizer.pkl

fakenewsmodel.pkl

5️⃣ Run the Streamlit App
streamlit run app.py

Open browser at:

http://localhost:8501


🧪 **Dataset**

Fake.csv – Fake news articles

True.csv – Real news articles

Each dataset contains:

Title

Text

Subject

Date


📌 **Use Cases**

News verification platforms

Media credibility analysis

NLP & ML learning projects

Resume & academic projects


📈 **Future Enhancements**

Deep Learning (LSTM / BERT)

API deployment (FastAPI)

Cloud hosting (Streamlit Cloud / Render)

User authentication & history tracking


👩‍💻 **Author**

Tejaswini Sajja
Final Year – Artificial Intelligence & Data Science
Passionate about ML, NLP & Full-Stack Development


⭐ **Acknowledgements**

Kaggle datasets

Scikit-learn documentation

Streamlit community


📜 **License**

This project is for educational purposes.
