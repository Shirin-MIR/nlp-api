


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class TextInput(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Analyse Reddit NLP</title>

    <style>
    button{
        background:#ff4500;
        color:white;
        border:none;
        padding:12px 25px;
        border-radius:8px;
        cursor:pointer;
        font-size:18px;
    }

    button:hover{
        background:#e03d00;
    }

    #resultat{
        margin-top:20px;
        font-size:30px;
        font-weight:bold;
    }
    </style>
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
                .innerHTML = data.sentiment;
        }
        </script>

    </body>
    </html>
    """


@app.post("/predict")
def predict(data: TextInput):
    X = vectorizer.transform([data.text])
    prediction = model.predict(X)[0]

    sentiment = "😊 Positif" if prediction == 1 else "😞 Négatif"

    return {
        "text": data.text,


        "sentiment": sentiment
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)