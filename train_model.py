import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
import joblib

df=pd.read_csv("sample_dataset.csv")
X_train,X_test,y_train,y_test=train_test_split(df["text"],df["label"],test_size=0.3,random_state=42)
model=Pipeline([
("tfidf",TfidfVectorizer(stop_words="english")),
("clf",LogisticRegression(max_iter=1000))
])
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
joblib.dump(model,"fake_news_model.pkl")
print("Model saved.")
