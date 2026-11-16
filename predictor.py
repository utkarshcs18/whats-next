import os
import re
import string
import pickle
import numpy as np

# Try importing TensorFlow / Keras if available
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    pass


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class NextWordPredictor:
    def __init__(
        self,
        model_path: str = "lstm_model.h5",
        tokenizer_path: str = "tokenizer.pkl",
        max_len_path: str = "max_len.pkl"
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_len_path = max_len_path
        
        self.model = None
        self.tokenizer = None
        self.max_len = 20
        self.index_to_word = {}
        self.is_loaded = False
        self.load_error = None
        
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            self.load_error = f"Model file '{self.model_path}' not found."
            return
        if not os.path.exists(self.tokenizer_path):
            self.load_error = f"Tokenizer file '{self.tokenizer_path}' not found."
            return
        if not TF_AVAILABLE:
            self.load_error = "TensorFlow is not installed in this Python environment."
            return

        try:
            # Load tokenizer
            with open(self.tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            
            # Build index_to_word mapping
            if hasattr(self.tokenizer, 'index_word'):
                self.index_to_word = self.tokenizer.index_word
            elif hasattr(self.tokenizer, 'word_index'):
                self.index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}

            # Load max_len
            if os.path.exists(self.max_len_path):
                with open(self.max_len_path, "rb") as f:
                    self.max_len = pickle.load(f)
            else:
                self.max_len = 20

            # Load Keras LSTM model
            self.model = load_model(self.model_path, compile=False)
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.load_error = f"Error loading model: {str(e)}"
            self.is_loaded = False

    def predict_top_k(self, text: str, top_k: int = 5) -> list[dict]:
        cleaned = clean_text(text)
        if not cleaned:
            return []

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            # Fallback to demo n-gram engine if model not trained yet
            return self._fallback_predict_top_k(cleaned, top_k)

        # Convert text to sequence
        seq = self.tokenizer.texts_to_sequences([cleaned])[0]
        if not seq:
            return []

        seq_padded = pad_sequences([seq], maxlen=self.max_len, padding='pre')

        # Predict probability distribution
        preds = self.model.predict(seq_padded, verbose=0)[0]

        # Get top-K indices
        top_indices = np.argsort(preds)[::-1][:top_k]

        results = []
        for idx in top_indices:
            word = self.index_to_word.get(idx, "")
            prob = float(preds[idx])
            if word and prob > 0:
                results.append({
                    "word": word,
                    "prob": prob,
                    "pct": f"{prob * 100:.1f}%"
                })
        return results

    def generate_sequence(self, seed_text: str, n_words: int = 5, temperature: float = 1.0) -> str:

        cleaned_seed = seed_text.strip()
        current_text = cleaned_seed

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return self._fallback_generate(cleaned_seed, n_words)

        for _ in range(n_words):
            cleaned = clean_text(current_text)
            seq = self.tokenizer.texts_to_sequences([cleaned])[0]
            if not seq:
                break
            seq_padded = pad_sequences([seq], maxlen=self.max_len, padding='pre')
            preds = self.model.predict(seq_padded, verbose=0)[0]

            if temperature <= 0.05:
                pred_index = int(np.argmax(preds))
            else:
                # Apply temperature sampling
                preds = np.asarray(preds).astype('float64')
                preds = np.log(np.maximum(preds, 1e-10)) / temperature
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                # Ensure sum == 1 for multinomial
                preds = preds / np.sum(preds)
                pred_index = int(np.random.choice(len(preds), p=preds))

            next_word = self.index_to_word.get(pred_index, "")
            if not next_word or next_word == "<unk>":
                break
            current_text += " " + next_word

        return current_text

    # --- Built-in Fallback Demo Engine for Instant Testing before training ---
    def _fallback_predict_top_k(self, cleaned: str, top_k: int = 5) -> list[dict]:
        words = cleaned.split()
        last_word = words[-1] if words else ""
        last_two = " ".join(words[-2:]) if len(words) >= 2 else last_word

        sample_knowledge = {
            "what are you": [("doing", 0.42), ("waiting", 0.28), ("thinking", 0.15), ("planning", 0.09), ("searching", 0.06)],
            "are you a": [("dreamer", 0.35), ("leader", 0.25), ("believer", 0.20), ("person", 0.12), ("creator", 0.08)],
            "the only": [("way", 0.45), ("limit", 0.25), ("person", 0.15), ("thing", 0.10), ("journey", 0.05)],
            "only way": [("to", 0.70), ("is", 0.15), ("forward", 0.10), ("out", 0.05)],
            "way to": [("get", 0.35), ("do", 0.30), ("live", 0.20), ("achieve", 0.15)],
            "life is": [("what", 0.38), ("not", 0.24), ("short", 0.18), ("beautiful", 0.12), ("simple", 0.08)],
            "success is": [("not", 0.45), ("the", 0.25), ("a", 0.18), ("peace", 0.12)],
            "believe in": [("yourself", 0.50), ("the", 0.25), ("your", 0.15), ("magic", 0.10)],
            "dreams": [("come", 0.40), ("and", 0.30), ("can", 0.20), ("are", 0.10)],
            "live": [("your", 0.35), ("a", 0.25), ("in", 0.20), ("every", 0.20)],
            "make": [("it", 0.30), ("each", 0.25), ("a", 0.25), ("others", 0.20)],
            "you": [("can", 0.30), ("are", 0.25), ("will", 0.20), ("have", 0.15), ("only", 0.10)],
            "the": [("future", 0.25), ("way", 0.20), ("world", 0.20), ("best", 0.20), ("greatest", 0.15)],
            "to": [("be", 0.30), ("get", 0.25), ("see", 0.20), ("live", 0.15), ("do", 0.10)],
            "in": [("life", 0.30), ("this", 0.25), ("the", 0.25), ("order", 0.20)]
        }

        # Check last two words, then last word, else default
        matches = sample_knowledge.get(last_two) or sample_knowledge.get(last_word)
        if not matches:
            matches = [("to", 0.30), ("and", 0.25), ("the", 0.20), ("a", 0.15), ("is", 0.10)]

        results = []
        for word, prob in matches[:top_k]:
            results.append({
                "word": word,
                "prob": prob,
                "pct": f"{prob * 100:.1f}%"
            })
        return results

    def _fallback_generate(self, seed: str, n_words: int = 5) -> str:
        current = seed
        for _ in range(n_words):
            top = self._fallback_predict_top_k(clean_text(current), top_k=1)
            if not top:
                break
            current += " " + top[0]["word"]
        return current
