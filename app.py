from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
from huggingface_hub import hf_hub_download

import gradio as gr
import uvicorn

# Download model
model_path = hf_hub_download(
    repo_id="WarTitan2077/Speed-Hit-Randomized",
    filename="speed_hit_model.pkl"
)
model = joblib.load(model_path)

# --- FastAPI app ---
app = FastAPI()

class Stats(BaseModel):
    player_speed: int
    player_weight: int
    player_attack_accuracy: float
    player_hit_accuracy: float
    player_avoidance: float
    enemy_speed: int
    enemy_weight: int
    enemy_attack_accuracy: float = 0.5
    enemy_hit_accuracy: float = 1.0
    enemy_avoidance: float = 0.5

@app.get("/")
def root():
    return RedirectResponse(url="/gradio")

@app.post("/predict")
def predict(stats: Stats):
    pbs = stats.player_speed - stats.player_weight
    ebs = stats.enemy_speed - stats.enemy_weight
    input_data = [[
        stats.player_speed, stats.player_weight, pbs,
        stats.player_attack_accuracy, stats.player_hit_accuracy, stats.player_avoidance,
        stats.enemy_speed, stats.enemy_weight, ebs,
        stats.enemy_attack_accuracy, stats.enemy_hit_accuracy, stats.enemy_avoidance
    ]]
    outcome = model.predict(input_data)[0]
    return {"outcome": outcome}

# --- Gradio interface ---
def gradio_predict(ps, pw, paa, pha, pao, es, ew, eaa, eha, eao):
    pbs = ps - pw
    ebs = es - ew
    input_data = [[ps, pw, pbs, paa, pha, pao, es, ew, ebs, eaa, eha, eao]]
    return model.predict(input_data)[0]

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Slider(10, 100, step=1, label="Player Speed"),
        gr.Slider(0, 10, step=1, label="Player Weight"),
        gr.Slider(0, 1, step=0.01, label="Player Attack Accuracy"),
        gr.Slider(0, 1.5, step=0.01, label="Player Hit Accuracy"),
        gr.Slider(0, 1, step=0.01, label="Player Avoidance"),
        gr.Slider(10, 100, step=1, label="Enemy Speed"),
        gr.Slider(0, 10, step=1, label="Enemy Weight"),
        gr.Slider(0, 1, step=0.01, label="Enemy Attack Accuracy"),
        gr.Slider(0, 1.5, step=0.01, label="Enemy Hit Accuracy"),
        gr.Slider(0, 1, step=0.01, label="Enemy Avoidance")
    ],
    outputs="text",
    title="Battle Outcome Predictor"
)

app = gr.mount_gradio_app(app, demo, path="/gradio")

# Only needed locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
