from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os
import gradio as gr

model_path = hf_hub_download(
    repo_id="WarTitan2077/Speed-Hit-Randomized",
    filename="speed_hit_model.pkl"
    token="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"
)

# Load the model
model = joblib.load(model_path)

# Label mapping
label_reverse = {
    0: "Player attacks twice and counters twice",
    1: "Enemy attacks twice and counters twice",
    2: "Both attack once"
}

# ---------- FastAPI API ----------
app = FastAPI()

class BattleStats(BaseModel):
    player_speed: int
    player_weight: int
    player_attack_accuracy: float
    player_hit_accuracy: float
    player_avoidance: float
    enemy_speed: int
    enemy_weight: int
    enemy_attack_accuracy: float
    enemy_hit_accuracy: float
    enemy_avoidance: float

@app.post("/predict")
def predict_api(stats: BattleStats):
    df = prepare_dataframe(stats)
    pred = model.predict(df)[0]
    return {"outcome": label_reverse[pred]}

def prepare_dataframe(stats):
    player_base_speed = stats.player_speed - stats.player_weight
    enemy_base_speed = stats.enemy_speed - stats.enemy_weight
    player_hit_chance = stats.player_attack_accuracy * stats.player_hit_accuracy * (1 - stats.enemy_avoidance)
    enemy_hit_chance = stats.enemy_attack_accuracy * stats.enemy_hit_accuracy * (1 - stats.player_avoidance)

    return pd.DataFrame([{
        "Player Speed": stats.player_speed,
        "Player Weight": stats.player_weight,
        "Player Base Speed": player_base_speed,
        "Player Attack Accuracy": stats.player_attack_accuracy,
        "Player Hit Accuracy": stats.player_hit_accuracy,
        "Player Avoidance": stats.player_avoidance,
        "Player Hit Chance": player_hit_chance,
        "Enemy Speed": stats.enemy_speed,
        "Enemy Weight": stats.enemy_weight,
        "Enemy Base Speed": enemy_base_speed,
        "Enemy Attack Accuracy": stats.enemy_attack_accuracy,
        "Enemy Hit Accuracy": stats.enemy_hit_accuracy,
        "Enemy Avoidance": stats.enemy_avoidance,
        "Enemy Hit Chance": enemy_hit_chance,
    }])

# ---------- Gradio Interface ----------
def predict_gradio(player_speed, player_weight, player_attack_accuracy, player_hit_accuracy, player_avoidance,
                   enemy_speed, enemy_weight, enemy_attack_accuracy, enemy_hit_accuracy, enemy_avoidance):
    
    stats = BattleStats(
        player_speed=player_speed,
        player_weight=player_weight,
        player_attack_accuracy=player_attack_accuracy,
        player_hit_accuracy=player_hit_accuracy,
        player_avoidance=player_avoidance,
        enemy_speed=enemy_speed,
        enemy_weight=enemy_weight,
        enemy_attack_accuracy=enemy_attack_accuracy,
        enemy_hit_accuracy=enemy_hit_accuracy,
        enemy_avoidance=enemy_avoidance
    )
    df = prepare_dataframe(stats)
    pred = model.predict(df)[0]
    return label_reverse[pred]

gradio_ui = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Slider(10, 100, label="Player Speed"),
        gr.Slider(0, 10, label="Player Weight"),
        gr.Slider(0, 1, step=0.01, label="Player Attack Accuracy"),
        gr.Slider(0, 1.5, step=0.01, label="Player Hit Accuracy"),
        gr.Slider(0, 1, step=0.01, label="Player Avoidance"),
        gr.Slider(10, 100, label="Enemy Speed"),
        gr.Slider(0, 10, label="Enemy Weight"),
        gr.Slider(0, 1, step=0.01, label="Enemy Attack Accuracy"),
        gr.Slider(0, 1.5, step=0.01, label="Enemy Hit Accuracy"),
        gr.Slider(0, 1, step=0.01, label="Enemy Avoidance"),
    ],
    outputs=gr.Text(label="Predicted Outcome"),
    title="Speed-Hit Battle Predictor"
)

# Mount Gradio at root ("/")
app.mount("/", WSGIMiddleware(gradio_ui.launch(inline=False)))
