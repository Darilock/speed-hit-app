from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("speed_hit_model.pkl")

# Label decoding
label_reverse = {0: "Player attacks twice and counters twice", 
                 1: "Enemy attacks twice and counters twice", 
                 2: "Both attack once"}

# Define API input structure
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

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(stats: BattleStats):
    # Compute derived features
    player_base_speed = stats.player_speed - stats.player_weight
    enemy_base_speed = stats.enemy_speed - stats.enemy_weight
    player_hit_chance = stats.player_attack_accuracy * stats.player_hit_accuracy * (1 - stats.enemy_avoidance)
    enemy_hit_chance = stats.enemy_attack_accuracy * stats.enemy_hit_accuracy * (1 - stats.player_avoidance)

    # Create DataFrame
    features = pd.DataFrame([{
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
        "Enemy Hit Chance": enemy_hit_chance
    }])

    pred = model.predict(features)[0]
    return {"outcome": label_reverse[pred]}
