import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


file_path = "data_train.jsonl"

# JSONL 파일을 읽음
data = pd.read_json(file_path, encoding='utf-8', lines=True)

# 텍스트 데이터를 리스트로 변환
questions = data["question"].astype(str).tolist()
answers = data["answer"].astype(str).tolist()

# 데이터 전처리: <start>와 <end> 토큰 추가
questions = ["<start> " + q + " <end>" for q in questions]
answers = ["<start> " + a + " <end>" for a in answers]

# <start> 바로 다음에 <end>가 오는 데이터 및 빈 답변 제거
filtered_questions = []
filtered_answers = []
for q, a in zip(questions, answers):
    if not (a.strip() == "<start> <end>" or a.strip() == ""):  # 빈 답변도 필터링
        filtered_questions.append(q)
        filtered_answers.append(a)

questions, answers = filtered_questions, filtered_answers

# 토크나이저 생성 및 저장
special_tokens = ["<start>", "<end>"]
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token="<unk>")
tokenizer.fit_on_texts(special_tokens + questions + answers)

# 토크나이저 저장
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# 단어 개수
vocab_size = len(tokenizer.word_index) + 1

max_question_length = max([len(q.split()) for q in questions])
max_answer_length = max([len(a.split()) for a in answers])

# 텍스트 길이 중 최대값을 max_len으로 설정
max_len = max(max_question_length, max_answer_length)
latent_dim = 32  # 모델 축소: latent_dim을 64에서 32로 줄임
embedding_dim = 32  # 모델 축소: embedding_dim을 64에서 32로 줄임

# 데이터 토큰화 및 패딩
input_sequences = tokenizer.texts_to_sequences(questions)
output_sequences = tokenizer.texts_to_sequences(answers)

input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len, padding='post')
output_data = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, maxlen=max_len, padding='post')

# 인코더 모델 정의 (GRU 층 수 줄임)
encoder_inputs = tf.keras.Input(shape=(max_len,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_gru = tf.keras.layers.GRU(latent_dim, return_state=True, dropout=0.2)
encoder_outputs, state_h = encoder_gru(encoder_embedding)
encoder_states = [state_h]

# 디코더 모델 정의 (GRU 층 수 줄임)
decoder_inputs = tf.keras.Input(shape=(max_len,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.2)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델 생성 및 컴파일
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 데이터셋 준비
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(output_data[:, :-1], maxlen=max_len, padding='post')
decoder_target_data = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(output_data[:, 1:], maxlen=max_len, padding='post'), -1)

# 배치 처리를 위한 데이터셋 생성
batch_size = 2
steps_per_epoch = len(input_data) // batch_size  # 한 에폭 당 스텝 수

# 데이터셋을 점차적으로 증가시키는 학습 로직
initial_size = 194098
increment_size = 1
max_size = len(input_data)  # 최대 데이터셋 크기 (전체 데이터)

for size in range(initial_size, max_size + 1, increment_size):
    print(f"학습 데이터 크기: {size}")
    
    # 부분 데이터셋을 선택
    input_data_batch = input_data[:size]
    decoder_input_data_batch = decoder_input_data[:size]
    decoder_target_data_batch = decoder_target_data[:size]
    
    # 배치 학습
    for step in range(steps_per_epoch):
        batch_start = step * batch_size
        batch_end = (step + 1) * batch_size
        input_batch = input_data_batch[batch_start:batch_end]
        decoder_input_batch = decoder_input_data_batch[batch_start:batch_end]
        decoder_target_batch = decoder_target_data_batch[batch_start:batch_end]

        # 모델 학습 (단계별로 학습)
        history = model.fit(
            [input_batch, decoder_input_batch], decoder_target_batch,
            epochs=1,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint('seq2seq_model_best.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)
            ]
        )

    # 모델 저장 (매 단계마다 모델을 저장할 수 있음)
    model.save(f"seq2seq_model_{size}.h5")
    print(f"모델 {size} 저장 완료")
