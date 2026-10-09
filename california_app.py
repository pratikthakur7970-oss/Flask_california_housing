from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import uvicorn

# Initialize app
app = FastAPI()


classifier = joblib.load("california2.pkl")


@app.get("/", response_class=HTMLResponse)
def main_page():

    return """
    <!DOCTYPE html>
    <html lang="en">

    <head>

        <meta charset="UTF-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>California Housing Price Predictor</title>

        <style>

            body {
                margin: 0;
                padding: 40px;
                font-family: Arial, sans-serif;
                background: linear-gradient(
                    135deg,
                    #0f2027,
                    #203a43,
                    #2c5364
                );
                color: white;
            }

            .container {
                max-width: 1000px;
                margin: auto;
            }

            h1 {
                text-align: center;
                margin-bottom: 35px;
            }

            .container-box {
                display: flex;
                gap: 30px;
                justify-content: center;
                flex-wrap: wrap;
            }

            .box {
                background: white;
                color: #222;
                width: 360px;
                padding: 25px;
                border-radius: 16px;
                box-shadow:
                    0 15px 35px rgba(0,0,0,0.3);
            }

            .box h2 {
                color: #203a43;
                margin-top: 0;
            }

            input {
                width: 100%;
                padding: 12px;
                margin: 7px 0;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-sizing: border-box;
            }

            button {
                width: 100%;
                padding: 13px;
                margin-top: 12px;
                border: none;
                border-radius: 8px;
                background: #203a43;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #0f2027;
            }

            .result {
                margin-top: 15px;
                padding: 12px;
                background: #eeeeee;
                border-radius: 8px;
                font-weight: bold;
            }

        </style>

    </head>


    <body>

        <div class="container">

            <h1>
                🏠 California Housing Price Predictor
            </h1>


            <div class="container-box">


                <div class="box">

                    <h2>
                        Predict from Input
                    </h2>


                    <input
                        type="number"
                        step="0.01"
                        id="MedInc"
                        placeholder="Median Income"
                    >


                    <input
                        type="number"
                        step="0.01"
                        id="HouseAge"
                        placeholder="House Age"
                    >


                    <input
                        type="number"
                        step="0.01"
                        id="AveRooms"
                        placeholder="Average Rooms"
                    >


                    <input
                        type="number"
                        step="0.01"
                        id="Population"
                        placeholder="Population"
                    >


                    <input
                        type="number"
                        step="0.01"
                        id="AveOccup"
                        placeholder="Average Occupancy"
                    >


                    <input
                        type="number"
                        step="0.01"
                        id="Latitude"
                        placeholder="Latitude"
                    >


                    <button onclick="predict()">
                        Predict
                    </button>


                    <div
                        class="result"
                        id="result"
                    >
                        Prediction will appear here
                    </div>

                </div>



                <div class="box">

                    <h2>
                        Predict from CSV File
                    </h2>


                    <input
                        type="file"
                        id="csvFile"
                        accept=".csv"
                    >


                    <button onclick="predictFile()">
                        Upload & Predict
                    </button>


                    <div
                        class="result"
                        id="fileResult"
                    >
                        CSV predictions will appear here
                    </div>

                </div>


            </div>

        </div>


        <script>


            async function predict() {

                const MedInc =
                    parseFloat(
                        document.getElementById("MedInc").value
                    );

                const HouseAge =
                    parseFloat(
                        document.getElementById("HouseAge").value
                    );

                const AveRooms =
                    parseFloat(
                        document.getElementById("AveRooms").value
                    );

                const Population =
                    parseFloat(
                        document.getElementById("Population").value
                    );

                const AveOccup =
                    parseFloat(
                        document.getElementById("AveOccup").value
                    );

                const Latitude =
                    parseFloat(
                        document.getElementById("Latitude").value
                    );


                const response =
                    await fetch(
                        `/predict?MedInc=${MedInc}&HouseAge=${HouseAge}&AveRooms=${AveRooms}&Population=${Population}&AveOccup=${AveOccup}&Latitude=${Latitude}`
                    );


                const data =
                    await response.json();


                document.getElementById(
                    "result"
                ).innerText =
                    "Predicted Price: " +
                    data.prediction.toFixed(2);

            }



            async function predictFile() {

                const fileInput =
                    document.getElementById("csvFile");

                const file =
                    fileInput.files[0];


                const formData =
                    new FormData();

                formData.append(
                    "file",
                    file
                );


                const response =
                    await fetch(
                        "/predict_file",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                document.getElementById(
                    "fileResult"
                ).innerText =
                    "Predictions: " +
                    data.predictions.join(", ");

            }

        </script>

    </body>

    </html>
    """


# Predict from query parameters

@app.get("/predict")
def predict(MedInc: float, HouseAge: float, AveRooms: float,
            Population: float, AveOccup: float, Latitude: float):

    input_data = [
        [
            MedInc,
            HouseAge,
            AveRooms,
            Population,
            AveOccup,
            Latitude
        ]
    ]

    prediction = classifier['model'].predict(input_data)

    return {
        "prediction": float(prediction[0])
    }


# Predict from uploaded CSV file

@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):

    df_test = pd.read_csv(file.file)

    prediction = classifier['model'].predict(df_test)

    return {
        "predictions": prediction.tolist()
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )