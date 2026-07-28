import joblib

model=joblib.load("fake_news_model.pkl")
print("Fake News Detection System")
while True:
    text=input("\nEnter news (or 'exit'): ")
    if text.lower()=="exit":
        break
    print("Prediction:",model.predict([text])[0])
