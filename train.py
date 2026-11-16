"""
Deep Learning Next Word Predictor - LSTM Training Pipeline
Trains an LSTM model on text quotes and saves artifacts:
- lstm_model.h5
- tokenizer.pkl
- max_len.pkl
- model_metadata.json
- training_curve.png
"""

import os
import sys
import json
import string
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    print("\n[ERROR] TensorFlow is not installed in this environment.")
    print("Please install it with: pip install tensorflow")
    print("Or run this script in Google Colab (free GPU available!).\n")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM Next Word Predictor")
    parser.add_argument("--data", type=str, default="quote_dataset.csv", help="Path to quotes CSV file")
    parser.add_argument("--col", type=str, default=None, help="Column name containing text/quotes (auto-detected if None)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Word embedding dimensions")
    parser.add_argument("--lstm_units", type=int, default=150, help="LSTM hidden units")
    parser.add_argument("--model_out", type=str, default="lstm_model.h5", help="Saved model file")
    parser.add_argument("--tokenizer_out", type=str, default="tokenizer.pkl", help="Saved tokenizer file")
    parser.add_argument("--max_len_out", type=str, default="max_len.pkl", help="Saved max_len file")
    return parser.parse_args()


def load_dataset(filepath: str, col_name: str = None) -> pd.Series:
    """Loads CSV and extracts the quotes/text column."""
    if not os.path.exists(filepath):
        if os.path.exists("sample_data.csv"):
            print(f"[INFO] '{filepath}' not found. Falling back to 'sample_data.csv'.")
            filepath = "sample_data.csv"
        else:
            raise FileNotFoundError(f"Dataset file '{filepath}' could not be found.")

    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset from '{filepath}' with shape {df.shape}")

    if col_name and col_name in df.columns:
        quotes = df[col_name].dropna().astype(str)
    else:
        # Auto-detect text column
        candidate_cols = ["quote", "quotes", "text", "sentence", "Quote", "Text"]
        found_col = None
        for c in candidate_cols:
            if c in df.columns:
                found_col = c
                break
        if not found_col:
            # Pick first string / object column
            object_cols = df.select_dtypes(include=["object"]).columns
            found_col = object_cols[0] if len(object_cols) > 0 else df.columns[0]
        
        print(f"[INFO] Auto-selected column: '{found_col}'")
        quotes = df[found_col].dropna().astype(str)

    return quotes


def preprocess_text(quotes_series: pd.Series) -> list[str]:
    """Cleans text: lowercase, punctuation removal, trim whitespace."""
    translator = str.maketrans('', '', string.punctuation)
    cleaned = []
    for line in quotes_series:
        text = line.lower().translate(translator).strip()
        if text:
            cleaned.append(text)
    print(f"[INFO] Preprocessed {len(cleaned)} clean sentences.")
    return cleaned


def create_sequences(cleaned_texts: list[str], max_vocab: int = 10000):
    """
    Fits tokenizer and creates n-gram sequences for next-word prediction.
    X: [word_1, word_2, ..., word_i]
    y: [word_{i+1}]
    """
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<unk>")
    tokenizer.fit_on_texts(cleaned_texts)

    # Note: total vocabulary size includes 0 (padding) + 1 to len(word_index)
    total_words = min(len(tokenizer.word_index) + 1, max_vocab)
    print(f"[INFO] Vocabulary size: {total_words} unique tokens (including padding)")

    sequences = tokenizer.texts_to_sequences(cleaned_texts)

    X, y = [], []
    for seq in sequences:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])

    print(f"[INFO] Generated {len(X)} training sequence pairs.")

    max_len = max([len(x) for x in X])
    print(f"[INFO] Max input sequence length: {max_len}")

    # Pad sequences with leading zeros ('pre')
    X_padded = pad_sequences(X, maxlen=max_len, padding='pre')

    # Convert y to categorical one-hot encoding
    y = np.array(y)
    y_categorical = to_categorical(y, num_classes=total_words)

    return X_padded, y_categorical, tokenizer, max_len, total_words


def build_lstm_model(vocab_size: int, max_len: int, embedding_dim: int = 100, lstm_units: int = 150):
    """Builds Sequential Deep Learning LSTM architecture."""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            name="embedding_layer"
        ),
        SpatialDropout1D(0.2, name="spatial_dropout"),
        LSTM(
            units=lstm_units,
            return_sequences=False,
            dropout=0.2,
            name="lstm_layer"
        ),
        Dense(units=vocab_size, activation="softmax", name="output_softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


def plot_and_save_history(history, filename="training_curve.png"):
    """Plots and saves loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', color='#4F46E5', lw=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss', color='#EF4444', lw=2, linestyle='--')
    ax1.set_title('Loss Curve', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Categorical Crossentropy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='#10B981', lw=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='#F59E0B', lw=2, linestyle='--')
    ax2.set_title('Accuracy Curve', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"[INFO] Training plots saved to '{filename}'")
    plt.close()


def test_prediction(model, tokenizer, max_len, seed_text="what are you"):
    """Runs a test prediction on the freshly trained model."""
    print(f"\n--- Model Test ---")
    print(f"Seed prompt: '{seed_text}'")

    cleaned = seed_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    seq = tokenizer.texts_to_sequences([cleaned])[0]
    seq_padded = pad_sequences([seq], maxlen=max_len, padding='pre')

    pred_probs = model.predict(seq_padded, verbose=0)[0]
    top_indices = np.argsort(pred_probs)[::-1][:3]

    print("Top 3 Predicted Next Words:")
    for rank, idx in enumerate(top_indices, start=1):
        word = tokenizer.index_word.get(idx, "<unk>")
        prob = pred_probs[idx] * 100
        print(f"  {rank}. {word} ({prob:.2f}%)")


def main():
    args = parse_args()

    print("=" * 60)
    print("  WHAT'S NEXT - LSTM NEXT WORD PREDICTOR TRAINING")
    print("=" * 60)

    # 1. Load Data
    quotes_series = load_dataset(args.data, args.col)

    # 2. Preprocess
    cleaned_texts = preprocess_text(quotes_series)

    # 3. Create Sequences
    X, y, tokenizer, max_len, vocab_size = create_sequences(cleaned_texts)

    # 4. Build Model
    model = build_lstm_model(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units
    )

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    # 6. Fit Model
    print(f"\n[INFO] Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Save Artifacts
    print(f"\n[INFO] Saving model to '{args.model_out}'...")
    model.save(args.model_out)

    print(f"[INFO] Saving tokenizer to '{args.tokenizer_out}'...")
    with open(args.tokenizer_out, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"[INFO] Saving max_len to '{args.max_len_out}'...")
    with open(args.max_len_out, "wb") as f:
        pickle.dump(max_len, f)

    # Save metadata JSON for the frontend to inspect
    metadata = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "epochs_trained": len(history.history['loss']),
        "final_loss": round(float(history.history['loss'][-1]), 4),
        "final_accuracy": round(float(history.history['accuracy'][-1]), 4),
        "embedding_dim": args.embedding_dim,
        "lstm_units": args.lstm_units
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 8. Save Plot
    plot_and_save_history(history, "training_curve.png")

    # 9. Test sample inference
    test_prediction(model, tokenizer, max_len, "what are you")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE! ALL ARTIFACTS SAVED:")
    print(f"  - Model:     {args.model_out}")
    print(f"  - Tokenizer: {args.tokenizer_out}")
    print(f"  - Max Length:{args.max_len_out}")
    print(f"  - Metadata:  model_metadata.json")
    print(f"  - Curves:    training_curve.png")
    print("=" * 60)
    print("\nRun the frontend now using: streamlit run app.py\n")


if __name__ == "__main__":
    main()
