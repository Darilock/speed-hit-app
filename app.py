import pandas as pd
import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import gradio as gr
import os

token = os.getenv("HF_TOKEN")  # safe access via environment variable


model_path = hf_hub_download(
    repo_id="WarTitan2077/Speed-Hit-Randomized",
    filename="speed_hit_model.pkl",
    token=token  # will be None if not set, but okay for public models
)

# Load the model
model = joblib.load(model_path)

# Initialize FastAPI
app = FastAPI()

# Define Pydantic model for API input
class BattleInput(BaseModel):
    player_speed: int
    player_weight: int
    enemy_speed: int
    enemy_weight: int
    player_attack_accuracy: float
    player_hit_accuracy: float
    player_avoidance: float
    enemy_attack_accuracy: float
    enemy_hit_accuracy: float
    enemy_avoidance: float

# Calculate derived features
def prepare_input(data: dict) -> pd.DataFrame:
    ps = data["player_speed"]
    pw = data["player_weight"]
    es = data["enemy_speed"]
    ew = data["enemy_weight"]

    paa = data["player_attack_accuracy"]
    pha = data["player_hit_accuracy"]
    pao = data["player_avoidance"]

    eaa = data["enemy_attack_accuracy"]
    eha = data["enemy_hit_accuracy"]
    eao = data["enemy_avoidance"]

    player_base_speed = ps - pw
    enemy_base_speed = es - ew

    player_hit_chance = paa * pha * (1 - eao)
    enemy_hit_chance = eaa * eha * (1 - pao)

    return pd.DataFrame([{
        "Player Speed": ps,
        "Player Weight": pw,
        "Player Base Speed": player_base_speed,
        "Player Attack Accuracy": paa,
        "Player Hit Accuracy": pha,
        "Player Avoidance": pao,
        "Player Hit Chance": round(player_hit_chance, 2),

        "Enemy Speed": es,
        "Enemy Weight": ew,
        "Enemy Base Speed": enemy_base_speed,
        "Enemy Attack Accuracy": eaa,
        "Enemy Hit Accuracy": eha,
        "Enemy Avoidance": eao,
        "Enemy Hit Chance": round(enemy_hit_chance, 2)
    }])

# FastAPI route
@app.post("/predict")
async def predict(input_data: BattleInput):
    input_df = prepare_input(input_data.dict())
    prediction = int(model.predict(input_df)[0])
    label = labels.get(prediction, "Unknown outcome")
    return {"outcome": label}

    
def gradio_predict(ps, pw, es, ew, paa, pha, pao, eaa, eha, eao):
    data = {
        "player_speed": ps,
        "player_weight": pw,
        "enemy_speed": es,
        "enemy_weight": ew,
        "player_attack_accuracy": paa,
        "player_hit_accuracy": pha,
        "player_avoidance": pao,
        "enemy_attack_accuracy": eaa,
        "enemy_hit_accuracy": eha,
        "enemy_avoidance": eao,
    }
    input_df = prepare_input(data)
    pred = model.predict(input_df)[0]
    labels = {
        0: "Player attacks twice and counters twice",
        1: "Enemy attacks twice and counters twice",
        2: "Both attack once"
}
    return labels.get(pred, "Unknown")


# Gradio UI
demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Slider(10, 100, label="Player Speed"),
        gr.Slider(0, 10, label="Player Weight"),
        gr.Slider(10, 100, label="Enemy Speed"),
        gr.Slider(0, 10, label="Enemy Weight"),
        gr.Slider(0.0, 1.0, step=0.01, label="Player Attack Accuracy"),
        gr.Slider(0.0, 1.5, step=0.01, label="Player Hit Accuracy"),
        gr.Slider(0.0, 1.0, step=0.01, label="Player Avoidance"),
        gr.Slider(0.0, 1.0, step=0.01, label="Enemy Attack Accuracy"),
        gr.Slider(0.0, 1.5, step=0.01, label="Enemy Hit Accuracy"),
        gr.Slider(0.0, 1.0, step=0.01, label="Enemy Avoidance"),
    ],
    outputs="text",
    title="Battle Outcome Predictor",
)

# Mount Gradio app on FastAPI
@app.get("/")
def gradio_root():
    return {"message": "Go to /gradio for the UI or POST to /predict for API access"}

# Make sure to do this *after* all FastAPI routes are declared
app = gr.mount_gradio_app(app, demo, path="/gradio")
