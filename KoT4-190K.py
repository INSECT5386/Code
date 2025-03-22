import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 1. JSONL 데이터 로드
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append((item["question"], item["answer"]))
    print(f"Loaded {len(data)} data points.")
    return data

# 2. PyTorch Dataset 정의
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=121):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]

        # 토크나이저 적용
        input_ids = self.tokenizer(question, self.max_length)
        target_ids = self.tokenizer(answer, self.max_length)

        # 디버깅: 데이터가 제대로 토크나이즈 되었는지 출력
        if idx < 5:  # 처음 몇 개 샘플만 출력
            print(f"Sample {idx}:")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Input IDs: {input_ids}")
            print(f"Target IDs: {target_ids}")
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

# 3. 토크나이저 클래스 정의
class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.word2idx = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2, "[EOS]": 3}
            self.idx2word = {v: k for k, v in self.word2idx.items()}
        else:
            self.word2idx = vocab
            self.idx2word = {v: k for k, v in vocab.items()}

    def build_vocab(self, sentences, min_freq=1):
        """주어진 문장 리스트에서 단어 빈도를 계산하여 어휘 구축"""
        word_freq = {}

        # 단어 빈도 계산
        for sentence in sentences:
            for word in sentence.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # 최소 빈도 기준으로 단어 추가
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_length=121):
        """문장을 토큰 ID 리스트로 변환"""
        tokens = [self.word2idx.get(word, 1) for word in text.split()]  # OOV는 [UNK] 처리
        tokens = [2] + tokens[: max_length - 2] + [3]  # [SOS], [EOS] 추가
        tokens += [0] * (max_length - len(tokens))  # 패딩 적용
        return tokens

    def decode(self, token_ids):
        """토큰 ID 리스트를 다시 문장으로 변환"""
        words = [self.idx2word.get(idx, "[UNK]") for idx in token_ids if idx > 3]  # 특수 토큰 제외
        return " ".join(words)

    def save_vocab(self, file_path="tokenizer_vocab.json"):
        """현재 어휘 사전을 JSON 파일로 저장"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)

    def load_vocab(self, file_path="tokenizer_vocab.json"):
        """JSON 파일에서 어휘 사전을 불러옴"""
        with open(file_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
            self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __call__(self, text, max_length=121):
        """토큰화를 편하게 하기 위해 __call__ 추가"""
        return self.encode(text, max_length)

# 4. 데이터 준비
data_path = "data_train.jsonl"  # 실제 jsonl 데이터 경로
dataset = load_jsonl(data_path)

# 5. 토크나이저 생성 및 어휘 구축
tokenizer = SimpleTokenizer()
all_sentences = [q for q, a in dataset] + [a for q, a in dataset]
tokenizer.build_vocab(all_sentences)

# 6. DataLoader 생성
train_dataset = ChatDataset(dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size, model_dim, n_heads, num_layers, max_length=121):
        super(Transformer, self).__init__()

        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, model_dim)
        
        # 위치 인코딩
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_length, model_dim))

        # Transformer Encoder 및 Decoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # 출력층 (예측을 위한 vocab_size 차원으로 변환)
        self.fc_out = nn.Linear(model_dim, vocab_size)

    def forward(self, src, tgt):
        seq_len_src = src.size(1)
        seq_len_tgt = tgt.size(1)

        # 위치 인코딩 적용
        pos_enc_src = self.pos_encoding[:, :seq_len_src, :]
        pos_enc_tgt = self.pos_encoding[:, :seq_len_tgt, :]

        # 임베딩 및 위치 인코딩 더하기
        src = self.embedding(src) + pos_enc_src
        tgt = self.embedding(tgt) + pos_enc_tgt

        # (batch, seq, dim) → (seq, batch, dim) 변환
        src = src.permute(1, 0, 2)  # [seq, batch, model_dim]
        tgt = tgt.permute(1, 0, 2)  # [seq, batch, model_dim]

        # Transformer Encoder
        memory = self.encoder(src)

        # Transformer Decoder
        output = self.decoder(tgt, memory)

        # (batch, seq, vocab) 형태로 변환하여 출력
        return self.fc_out(output.permute(1, 0, 2))  # [batch, seq, vocab_size]


# 8. 모델 초기화 및 학습 설정
vocab_size = len(tokenizer.word2idx)
model = Transformer(vocab_size, model_dim=256, n_heads=8, num_layers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. 학습 루프
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch_idx, batch in enumerate(train_loader):
        src, tgt = batch["input_ids"], batch["target_ids"]

        optimizer.zero_grad()

        # [EOS]를 예측하도록 학습
        outputs = model(src, tgt[:, :-1])  # [EOS] 제외
        outputs = outputs[:, :tgt.shape[1] - 1, :]  # 크기 맞추기

        # Loss 계산
        loss = criterion(outputs.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))

        # 배치마다 손실 출력
        if batch_idx % 10 == 0:  # 10번째 배치마다 출력
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 모델 학습 완료 후, 토크나이저 저장
tokenizer.save_vocab("tokenizer_vocab.json")

# 모델 저장
torch.save(model.state_dict(), 'transformer_model.pth')
