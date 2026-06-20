


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
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analyse Reddit NLP</title>
    </head>
    <body>
        <h1>Analyse des sentiments Reddit</h1>

        <textarea id="text" rows="5" cols="60"></textarea>
        <br><br>

        <button onclick="analyse()">Analyser</button>

        <h2 id="resultat"></h2>

        <script>
        async function analyse() {

            const texte =
                document.getElementById("text").value;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: texte
                })
            });

            const data = await response.json();

            document.getElementById("resultat")
                .innerHTML =
                "Prédiction : " + data.prediction;
        }
        </script>

    </body>
    </html>
    """


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