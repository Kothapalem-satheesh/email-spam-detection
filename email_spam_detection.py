

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import streamlit as st

# # ----------------------------
# # 🎯 App Title and Description
# # ----------------------------
# st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="centered")
# st.title("📧 Email Spam Detection App")
# st.write("""
# This app uses a **Naive Bayes Classifier** trained on SMS messages to detect whether a message is **Spam** or **Ham (Not Spam)**.
# """)

# # ----------------------------
# # 📂 Load Dataset
# # ----------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv(r"C:\Users\sathe\Downloads\archive (2)\spam.csv", encoding='latin-1')
#     df = df[['v1', 'v2']]
#     df.columns = ['label', 'message']
#     df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
#     return df

# df = load_data()

# # ----------------------------
# # 🧠 Prepare and Train Model
# # ----------------------------
# @st.cache_resource
# def train_model(df):
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['message'], df['label_num'], test_size=0.2, random_state=42
#     )

#     vectorizer = CountVectorizer()
#     tfidf_transformer = TfidfTransformer()

#     X_train_counts = vectorizer.fit_transform(X_train)
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#     model = MultinomialNB()
#     model.fit(X_train_tfidf, y_train)

#     # Evaluate model
#     X_test_counts = vectorizer.transform(X_test)
#     X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#     y_pred = model.predict(X_test_tfidf)

#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     class_report = classification_report(y_test, y_pred, output_dict=False)

#     return model, vectorizer, tfidf_transformer, accuracy, conf_matrix, class_report

# model, vectorizer, tfidf_transformer, accuracy, conf_matrix, class_report = train_model(df)

# # ----------------------------
# # 📊 Model Performance
# # ----------------------------
# with st.expander("📈 Model Performance Details"):
#     st.metric("Accuracy", f"{accuracy * 100:.2f}%")
#     st.write("**Confusion Matrix:**")
#     st.dataframe(pd.DataFrame(conf_matrix, columns=["Predicted Ham", "Predicted Spam"],
#                               index=["Actual Ham", "Actual Spam"]))
#     st.text("**Classification Report:**")
#     st.text(class_report)

# # ----------------------------
# # ✉️ User Input for Prediction
# # ----------------------------
# st.subheader("🔍 Check a Message")
# user_input = st.text_area("Enter a message to check if it's Spam or Ham:")

# if st.button("Predict"):
#     if user_input.strip():
#         input_counts = vectorizer.transform([user_input])
#         input_tfidf = tfidf_transformer.transform(input_counts)
#         prediction = model.predict(input_tfidf)
#         result = "🚨 Spam" if prediction[0] == 1 else "✅ Ham (Not Spam)"
#         st.success(f"**Result:** {result}")
#     else:
#         st.warning("Please enter a message before predicting.")

# # ----------------------------
# # 🧾 Footer
# # ----------------------------
# st.markdown("---")
# st.caption("Built with ❤️ using Streamlit by kothapalem satheesh")




import re
import nltk
import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# 🧩 Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="📧 Email Spam Detector", page_icon="📬", layout="centered")

# ----------------------------
# 🎨 Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #F9FAFC;
        padding: 2rem;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #4A90E2;
        text-align: center;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 10em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🧠 Title and Intro
# ----------------------------
st.title("📧 Email & SMS Spam Detection App")
# st.write("""
# This app uses **TF-IDF + Logistic Regression** to detect whether a message is **Spam** or **Ham (Not Spam)**.
# Trained on the popular **SMS Spam Collection Dataset**.
# """)

# ----------------------------
# 📂 Load and Preprocess Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\sathe\Downloads\archive (2)\spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# ----------------------------
# 🧹 Text Cleaning & Lemmatization
# ----------------------------
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove numbers/punctuations
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['cleaned'] = df['message'].apply(clean_text)

# ----------------------------
# ✂️ Split Dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label_num'], test_size=0.2, random_state=42
)

# ----------------------------
# 🔤 TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------
# 🤖 Model Training (Best Accuracy)
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ----------------------------
# 📈 Evaluation
# ----------------------------
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=False)

# ----------------------------
# 📊 Show Model Performance
# ----------------------------
with st.expander("📈 Model Performance Details"):
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
    st.write("**Confusion Matrix:**")
    st.dataframe(pd.DataFrame(conf_matrix, 
                              columns=["Predicted Ham", "Predicted Spam"],
                              index=["Actual Ham", "Actual Spam"]))
    st.text("**Classification Report:**")
    st.text(class_report)

# ----------------------------
# ✉️ User Input Prediction
# ----------------------------
st.subheader("🔍 Check a Message")

user_input = st.text_area("Enter a message to analyze:", placeholder="Type or paste a message here...")

if st.button("Predict"):
    if user_input.strip():
        cleaned_input = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_tfidf)
        if prediction[0] == 1:
            st.markdown('<p style="color:red; font-size:24px; text-align:center;">🚨 This message is **SPAM**!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green; font-size:24px; text-align:center;">✅ This message is **Not Spam**.</p>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message before predicting.")

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit by kothapalem satheesh")
