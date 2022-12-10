from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
async def root():
    return {"message": "Welcome to Iris ML API"}

@app.get("/predict/{sl}/{sw}/{pl}/{pw}")
async def predict(sl: float, sw: float, pl: float, pw: float):
    input = np.array([[sl, sw, pl, pw]])
    predicted = model.predict(input).tolist()
    return {'prediciton': predicted[0]}