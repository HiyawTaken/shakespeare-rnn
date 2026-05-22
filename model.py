import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

#file cleaned up
'''with open("shakespeare.txt", "r+", encoding="utf-8") as f:
    file = f.read()
    file = str(file).lower()
    keep = set('abcdefghijklmnopqrstuvwxyz \n.,!?;:-\'\"')
    with open("shakespeare.txt", "w") as c:
        for letter in file:
            if letter in keep:
                c.write(letter)'''

corpus_set = sorted('abcdefghijklmnopqrstuvwxyz \n.,!?;:-\'\"')
corpus_idx = {char: idx for idx, char in enumerate(corpus_set)}

idx_to_corpus = {v: k for k, v in corpus_idx.items()}

def char_to_index(corpus):
    output = []
    for letter in corpus.lower():
        if letter in corpus_idx:
            output.append(corpus_idx[letter])
    return output

def index_to_char(indexes):
    output = ""
    for idx in indexes:
        output += idx_to_corpus[idx]
    return output

def build_dataset(encoded_corpus, win_size):
    windows_x = [encoded_corpus[x: x + win_size] for x in range(0, len(encoded_corpus) - win_size)]
    targets = [encoded_corpus[x + win_size] for x in range(0, len(encoded_corpus) - win_size)]

    windows_x = np.array(windows_x, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    windows_x = windows_x.reshape(-1, win_size)

    return windows_x, targets

def main():
    #parameters
    window_size = 40
    vocab_size = len(corpus_idx)
    embedding_dim = 64

    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        corpus = f.read()

        encoded_corpus = char_to_index(corpus[:3000000])

        train_end = int(len(encoded_corpus) * 0.8)
        val_end = int(len(encoded_corpus) * 0.9)

        train_idx_corpus = encoded_corpus[:train_end]
        val_idx_corpus = encoded_corpus[train_end:val_end]
        test_idx_corpus = encoded_corpus[val_end:]

    X_train, y_train = build_dataset(train_idx_corpus, window_size)
    X_val, y_val = build_dataset(val_idx_corpus, window_size)
    X_test, y_test = build_dataset(test_idx_corpus, window_size)

    model = Sequential([
        Input(shape=(window_size,)),
        Embedding(vocab_size, embedding_dim),
        SimpleRNN(256, return_sequences=True),
        SimpleRNN(256, return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    checkpoint_callback = ModelCheckpoint(
        filepath='shakespeare_rnn.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint_callback]
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    save_model = input("Would you like to save the model(y/n)?")
    if save_model == "y":
        model.save('shakespeare_rnn_final.keras')
    else:
        exit()


if __name__ == '__main__':
    main()
