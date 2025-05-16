import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import sys

# ===== 1. 토크나이저 불러오기 =====
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

word_to_index = tokenizer_data["word_to_index"]
index_to_word = {int(k): v for k, v in tokenizer_data["index_to_word"].items()}

vocab_size = len(word_to_index)
pad_id = word_to_index["<pad>"]
end_id = word_to_index["<end>"]

def text_to_ids(text):
    return [word_to_index.get(tok, word_to_index["<unk>"]) for tok in text.split()]

def ids_to_text(ids):
    return " ".join([index_to_word.get(i, "<unk>") for i in ids])

# ===== 2. 모델 정의 =====
class GEGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = layers.Dense(d_ff * 2)
        self.out = layers.Dense(d_model)

    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.gelu(x_gate))

class GPTBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = GEGLU(d_model, d_ff)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        x_norm = self.ln1(x)
        attn_out = self.attn(query=x_norm, value=x_norm, key=x_norm, use_causal_mask=True)
        x = x + self.dropout1(attn_out, training=training)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out, training=training)
        return x

class GPT(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, d_model, d_ff, n_layers, num_heads=8):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=[seq_len, d_model],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
        )
        self.blocks = [GPTBlock(d_model, d_ff, num_heads) for _ in range(n_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.head = layers.Dense(vocab_size)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.token_embedding(x) + self.pos_embedding[tf.newaxis, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

# ===== 3. 생성 함수 (Top-p 샘플링) =====
def generate_text_topp(model, prompt, max_gen=98, p=0.9, temperature=0.8, min_len=20):
    model_input = text_to_ids(f"<start> {prompt}")
    model_input = model_input[:max_len]
    generated = list(model_input)

    for step in range(max_gen):
        input_padded = np.pad(generated, (0, max_len - len(generated)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded])
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(generated) - 1].numpy()

        next_token_logits[end_id] -= 5.0
        next_token_logits[pad_id] -= 10.0

        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)

        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)

        next_token_id = np.random.choice(top_indices, p=top_probs)

        if next_token_id == end_id and len(generated) >= min_len:
            break

        generated.append(next_token_id)

        sys.stdout.write(index_to_word.get(next_token_id, "<unk>") + ' ')
        sys.stdout.flush()

    print()  # 줄바꿈
    return ids_to_text(generated)

# ===== 4. 모델 로딩 및 실행 =====
max_len = 100
model = GPT(vocab_size=vocab_size, seq_len=max_len, d_model=100, d_ff=400, n_layers=4)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4))  # 필요 없어도 safety용
model.load_weights("VecterA.weights.h5")

# ===== 5. 생성 테스트 =====
prompt = "요즘 인기 많은 음식은 뭐야?"
print("\n[ VecterA의 응답 ]")
print(generate_text_topp(model, prompt, p=0.9))
