from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import keras
import numpy as np
from model import char_to_index, idx_to_corpus, corpus_idx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = keras.models.load_model('shakespeare_rnn.keras')

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

@app.get("/")
def serve_frontend():
    return FileResponse("static/frontend.html")

@app.post("/autocomplete/query")
async def autocomplete(query: str, temperature: float = 0.1):
    output = ""
    #loop to predict the next 20 characters
    for _ in range(20):
        #prepare data to be fed into the model
        query = query.lower()
        encoded_query = char_to_index(query)
        encoded_query = encoded_query[-40:]
        if len(encoded_query) < 40:
            encoded_query = [0] * (40 - len(encoded_query)) + encoded_query
        encoded_query = np.array(encoded_query)
        encoded_query = encoded_query.reshape(1, 40, 1)

        vector_prediction = model.predict(encoded_query, verbose=0)

        #adjust for temp
        vector_prediction = softmax(vector_prediction / temperature)
        vector_prediction = vector_prediction.flatten()
        vector_prediction = vector_prediction / np.sum(vector_prediction)

        prediction = idx_to_corpus[np.random.choice(len(vector_prediction), p=vector_prediction)]

        query += prediction
        output += prediction
    return {"output": output}
