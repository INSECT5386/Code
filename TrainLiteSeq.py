import numpy as np  
import pandas as pd  
import json  
from tqdm import tqdm  
import joblib  
from LiteSeq import SimpleEncoder, SimpleDecoder, Adam, Seq2Seq

optimizer = Adam

# 전처리 함수
def preprocess(text):
    if isinstance(text, str):  # text가 문자열일 경우에만 strip 호출
        return f"<start> {text.strip()} <end>"
    else:
        return ""  # 문자열이 아닌 경우 빈 문자열 반환

# 데이터셋 로드 (이미 변환된 CSV 파일 사용)
df = pd.read_csv("https://huggingface.co/datasets/Yuchan5386/MINDAI/resolve/main/data.csv?download=true")

# NaN 값이나 float 값을 처리하기 위한 방법
df['question'] = df['question'].fillna('')  # NaN 값을 빈 문자열로 처리
df['answer'] = df['answer'].fillna('')      # NaN 값을 빈 문자열로 처리

# 전처리 적용
questions = [preprocess(q) for q in df["question"]]
answers = [preprocess(str(a)) for a in df["answer"]]  # str()로 변환하여 처리

# 결과 확인
print(questions[:5])
print(answers[:5])
  
vocab = sorted(list(set(" ".join(questions + answers).split())))  
vocab_size = len(vocab)  
word2idx = {w: i for i, w in enumerate(vocab)}  
idx2word = {i: w for i, w in enumerate(vocab)}  
  
def encode_sequence(seq, word2idx):  
    return [word2idx[word] for word in seq.split() if word in word2idx]  
  
def pad_sequences(seqs, pad_value=0):  
    max_len = max(len(seq) for seq in seqs)  
    return np.array([seq + [pad_value] * (max_len - len(seq)) for seq in seqs])  
  
X = [encode_sequence(q, word2idx) for q in questions[:1000]]  
Y = [encode_sequence(a, word2idx) for a in answers[:1000]]  
X_padded = pad_sequences(X)  
Y_padded = pad_sequences(Y)  
  
tokenizer = {"word2idx": word2idx, "idx2word": {str(k): v for k, v in idx2word.items()}}  
with open("tokenizer.json", "w", encoding="utf-8") as f:  
    json.dump(tokenizer, f, ensure_ascii=False, indent=4)  
  
print("토크나이저 저장 완료!")  
 

embed_size, hidden_size = 64, 64
encoder = GRUEncoder(vocab_size, embed_size, hidden_size)  
decoder = GRUDecoder(vocab_size, embed_size, hidden_size)  
model = Seq2Seq(encoder, decoder, learning_rate=0.001)  


def train_on_batch(model, batch_x, batch_y):
    # 배치 단위로 학습
    batch_loss = 0
    for x, y in zip(batch_x, batch_y):
        loss = model.train_step(x, y)
        batch_loss += loss
    return batch_loss / len(batch_x)

batch_size = 32
epochs = 1

for epoch in range(epochs):
    epoch_loss = 0
    total_batches = (len(X_padded) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}", ncols=100, leave=True)

    for i in range(0, len(X_padded), batch_size):
        batch_x = X_padded[i:i + batch_size]
        batch_y = Y_padded[i:i + batch_size]

        # 학습
        batch_loss = train_on_batch(model, batch_x, batch_y)
        epoch_loss += batch_loss

        # 파라미터 업데이트
        model.update_params()

        # 진행바 업데이트
        avg_batch_loss = epoch_loss / ((i // batch_size) + 1)
        pbar.set_postfix({'Loss': f'{avg_batch_loss:.4f}'}, refresh=True)
        pbar.update(1)

    pbar.close()
    avg_loss = epoch_loss / total_batches
    print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

# -------------------- 모델 저장 -----------------  
joblib.dump(model, "Onlnp-LiteSeq.joblib")  
print("모델 저장 완료!")  
  
# -------------------- 모델 예측 --------------------  
def predict(model, input_text, word2idx, idx2word):  
    input_seq = encode_sequence(preprocess(input_text), word2idx)  
    prediction = model.predict(input_seq, word2idx, idx2word)  
    return prediction  
  
# 예시 예측  
input_text = "프랑스의 수도는 어디인가요?"  
output_text = predict(model, input_text, word2idx, idx2word)  
print(f"입력: {input_text}")  
print(f"예측: {output_text}")
