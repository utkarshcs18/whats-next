# ✨ What's Next? — Deep Learning LSTM Next-Word Predictor

A complete Deep Learning NLP application that predicts the most likely next word and generates multi-word continuations using an **LSTM (Long Short-Term Memory)** neural network. Packaged with a modern, aesthetic light-themed web interface built in Streamlit.

---

## 📁 Project Structure

```text
whats-next/
├── app.py               # Aesthetic light-themed Streamlit web app
├── predictor.py         # Model loading, Top-K inference, & sequence generator
├── train.py             # Full Python DL training pipeline (CLI configurable)
├── train_colab.ipynb    # Google Colab ready notebook for 1-click free GPU training
├── sample_data.csv      # Sample quotes dataset for instant local testing
├── requirements.txt     # Python package requirements
└── README.md            # Documentation and instructions
```

---

## 🧠 Deep Learning Architecture

The model processes text sequences as follows:

```
[Input Text] ➔ Tokenizer ➔ Pre-Padding (max_len)
                      ⬇
           Embedding Layer (100d)
                      ⬇
         SpatialDropout1D (0.2)
                      ⬇
            LSTM Layer (150 units)
                      ⬇
       Dense Layer + Softmax Activation
                      ⬇
    [Next Word Probability Distribution]
```

- **Embedding Layer**: Projects discrete word integers into a dense vector space capturing semantic relationships.
- **LSTM Layer**: Retains context and sequential word dependencies across temporal steps.
- **Softmax Output**: Computes probability distribution across all unique vocabulary tokens.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note: TensorFlow is required for training. The web frontend also has an interactive preview mode if you want to explore the UI before training).*

---

### 2. Manual Training (Choose Option A or B)

#### Option A: In Google Colab (Recommended for Free GPU)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `train_colab.ipynb`.
3. Upload your `qoute_dataset.csv` when prompted.
4. Run all cells — it will automatically train the model and download:
   - `lstm_model.h5`
   - `tokenizer.pkl`
   - `max_len.pkl`
5. Place these files into your `whats-next/` directory.

#### Option B: Train Locally via CLI
```bash
# Using your quote_dataset.csv
python train.py --data quote_dataset.csv --epochs 20 --batch_size 64

# Or test with the included sample data
python train.py --data sample_data.csv --epochs 15
```

This will automatically create:
- `lstm_model.h5`
- `tokenizer.pkl`
- `max_len.pkl`
- `model_metadata.json`
- `training_curve.png` (Loss & accuracy plots)

---

### 3. Launch the Aesthetic Light-Themed Web Frontend

Run the following command in your terminal:
```bash
streamlit run app.py
```

The app will open automatically in your browser (default: `http://localhost:8501`).

---

## 🎨 Frontend Features

- **Aesthetic Light Palette**: Soft gradients, modern typography, rounded glass cards, and clean drop shadows.
- **Top-5 Next Word Suggestions**: Shows top candidates with confidence percentages.
- **Click-to-Append Chips**: Click any predicted word pill to append it to your sentence.
- **Live Sequence Generator**: Sliders for target word length and temperature / creativity.
- **Probability Distribution**: Sleek horizontal probability bars showing ranking distribution.
- **Quick Inspiration Buttons**: One-click prompt starters (`The only way`, `Success is`, `Life is what`, etc.).
- **Model File Uploader**: Directly upload or swap `lstm_model.h5` and `tokenizer.pkl` through the sidebar.
