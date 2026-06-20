


from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class TextInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "API NLP OK"}


@app.post("/predict")
def predict(data: TextInput):
    X = vectorizer.transform([data.text])
    prediction = model.predict(X)[0]

    return {
        "text": data.text,
        "prediction": int(prediction)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)