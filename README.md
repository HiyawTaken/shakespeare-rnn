# Shakespeare RNN Autocomplete

A character-level language model built with a SimpleRNN trained on the complete works of Shakespeare. Type a seed phrase and the model generates a continuation one character at a time.

## Live Demo

https://shakespeare-rnn.onrender.com

## What it does

Enter a seed phrase in the browser. The model predicts the next character, appends it, and repeats — generating text character by character in the style of Shakespeare.

A temperature slider controls creativity. Low temperature produces conservative, repetitive output. High temperature produces unpredictable, creative output.

## How it works

- Architecture: 2-layer SimpleRNN (256 units each) → Dense (vocab_size) with Softmax
- Loss: Categorical cross entropy
- Optimizer: Adam
- Dataset: Complete works of Shakespeare (full corpus)
- Window size: 40 characters per input sequence
- Vocabulary: 50 unique characters
- Training: Converged at val_loss 1.59, val_accuracy 51%

The model reads a window of 40 characters, predicts a probability distribution over the vocabulary, samples from that distribution using temperature scaling, and appends the result. The window slides forward, and the process repeats.

## The Honest Result

The model generates non-sense. We have reached the structural ceiling of a SimpleRNN. The hidden state gets overwritten at every time step, erasing early context. This is the vanishing gradient problem made visible.

## Stack

- Python, NumPy, Keras, TensorFlow
- FastAPI backend
- Docker
- Vanilla HTML/CSS/JS frontend

## Run locally

```bash
docker build -t shakespeare-autocomplete .
docker run -p 8000:8000 shakespeare-autocomplete
```

Then open `http://localhost:8000`

## Project structure

```
├── main.py
├── model.py
├── shakespeare_rnn.keras
├── shakespeare.txt
├── static/
│   └── frontend.html
├── Dockerfile
└── requirements.txt
```
