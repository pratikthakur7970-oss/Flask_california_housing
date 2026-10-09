from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import uvicorn  # It is a server to run FastAPI

# Initialize app
app = FastAPI()   

# Load model
classifier = joblib.load("california2.pkl")

@app.get("/", response_class=HTMLResponse)
def main_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>California Housing Prediction</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #141e30, #243b55);
                margin: 0;
                padding: 40px;
                color: white;
            }

            .container {
                max-width: 900px;
                margin: auto;
            }

            h1 {
                text-align: center;
                margin-bottom: 30px;
            }

            .cards {
                display: flex;
                gap: 25px;
                justify-content: center;
                flex-wrap: wrap;
            }

            .card {
                background: white;
                color: #222;
                padding: 25px;
                border-radius: 15px;
                width: 350px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }

            .card h2 {
                margin-top: 0;
                color: #243b55;
            }

            input {
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                box-sizing: border-box;
                border: 1px solid #ccc;
                border-radius: 8px;
            }

            button {
                width: 100%;
                padding: 12px;
                margin-top: 10px;
                background: #243b55;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover {
                background: #141e30;
            }

            .result {
                margin-top: 15px;
                padding: 12px;
                background: #f1f1f1;
                border-radius: 8px;
                font-weight: bold;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>🏠 California Housing Price Predictor</h1>

            <div class="cards">

                <div class="card">

                    <h2>Manual Prediction</h2>

                    <input type="number" step="0.01" id="MedInc" placeholder="Median Income">

                    <input type="number" step="0.01" id="HouseAge" placeholder="House Age">

                    <input type="number" step="0.01" id="AveRooms" placeholder="Average Rooms">

                    <input type="number" step="0.01" id="Population" placeholder="Population">

                    <input type="number" step="0.01" id="AveOccup" placeholder="Average Occupancy">

                    <input type="number" step="0.01" id="Latitude" placeholder="Latitude">

                    <button onclick="predict()">Predict</button>

                    <div class="result" id="result">
                        Result will appear here
                    </div>

                </div>


                <div class="card">

                    <h2>CSV Prediction</h2>

                    <input type="file" id="csvFile" accept=".csv">

                    <button onclick="predictFile()">
                        Upload & Predict
                    </button>

                    <div class="result" id="fileResult">
                        CSV result will appear here
                    </div>

                </div>

            </div>

        </div>


        <script>

            async function predict() {

                const MedInc =
                    parseFloat(document.getElementById("MedInc").value);

                const HouseAge =
                    parseFloat(document.getElementById("HouseAge").value);

                const AveRooms =
                    parseFloat(document.getElementById("AveRooms").value);

                const Population =
                    parseFloat(document.getElementById("Population").value);

                const AveOccup =
                    parseFloat(document.getElementById("AveOccup").value);

                const Latitude =
                    parseFloat(document.getElementById("Latitude").value);


                const response = await fetch(
                    `/predict?MedInc=${MedInc}&HouseAge=${HouseAge}&AveRooms=${AveRooms}&Population=${Population}&AveOccup=${AveOccup}&Latitude=${Latitude}`
                );

                const data = await response.json();

                document.getElementById("result").innerText =
                    "Predicted Price: " + data.prediction.toFixed(2);
            }


            async function predictFile() {

                const fileInput =
                    document.getElementById("csvFile");

                const file =
                    fileInput.files[0];

                const formData =
                    new FormData();

                formData.append("file", file);


                const response = await fetch(
                    "/predict_file",
                    {
                        method: "POST",
                        body: formData
                    }
                );


                const data =
                    await response.json();


                document.getElementById("fileResult").innerText =
                    "Predictions: " +
                    data.predictions.join(", ");
            }

        </script>

    </body>
    </html>
    """


# Predict from query parameters

@app.get("/predict")
def predict(MedInc:float,HouseAge:float,AveRooms:float, 
           Population:float, AveOccup:float, Latitude:float):

    input_data = [[MedInc,HouseAge,AveRooms, 
           Population, AveOccup, Latitude]]

    prediction = classifier['model'].predict(input_data)

    return {"prediction": float(prediction[0])}


# Predict from uploaded CSV file

@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):

    df_test = pd.read_csv(file.file)

    prediction = classifier['model'].predict(df_test)

    return {"predictions": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)