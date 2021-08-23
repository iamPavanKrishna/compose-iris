import os
import uvicorn
import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from utils import process_data

TRAINR_ENDPOINT = os.getenv("TRAINR_ENDPOINT")

# defining the main app
app = FastAPI(title="processr", docs_url="/")

# class which is expected in the payload while training
class DataIn(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    non_flavanoids_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od: float
    proline: float
    wine_class: str


# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running.
def ping():
    return {"ping": "pong"}


@app.post("/process", status_code=200)
# Route to take in data, process it and send it for training.
def process(data: List[DataIn]):
    processed = process_data(data)
    # send the processed data to trainr for training
    response = requests.post(f"{TRAINR_ENDPOINT}/train", json=processed)
    return {"detail": "Processing successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on localhost:8888
    uvicorn.run("main:app", host="localhost", port=8888, reload=True)
