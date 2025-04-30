import numpy as np

class Adam:
    def __init__(self, params, grads, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def update(self):
        self.t += 1
        for i, param in enumerate(self.params):
            grad = self.grads[i]
            if grad is None: continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class GRUEncoder:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.embedding = np.random.randn(vocab_size, embed_size) * 0.01
        self.W_z = np.random.randn(embed_size + hidden_size, hidden_size) * 0.01
        self.W_r = np.random.randn(embed_size + hidden_size, hidden_size) * 0.01
        self.W_h = np.random.randn(embed_size + hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros(hidden_size)
        self.b_r = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)

        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_h = np.zeros_like(self.W_h)

    def forward(self, x):
        self.h = np.zeros(self.b_z.shape)
        self.h_list = []
        self.xs = []

        for idx in x:
            x_t = self.embedding[idx]
            self.xs.append(x_t)
            xh = np.concatenate([x_t, self.h])
            z = self.sigmoid(np.dot(xh, self.W_z) + self.b_z)
            r = self.sigmoid(np.dot(xh, self.W_r) + self.b_r)
            rh = r * self.h
            xrh = np.concatenate([x_t, rh])
            h_hat = np.tanh(np.dot(xrh, self.W_h) + self.b_h)
            self.h = (1 - z) * self.h + z * h_hat
            self.h_list.append(self.h.copy())

        return np.stack(self.h_list)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, dhs):
        self.dW_z.fill(0)
        self.dW_r.fill(0)
        self.dW_h.fill(0)
        db_z = np.zeros_like(self.b_z)
        db_r = np.zeros_like(self.b_r)
        db_h = np.zeros_like(self.b_h)

        dh_next = np.zeros_like(self.h)
        for t in reversed(range(len(self.xs))):
            x = self.xs[t]
            h = self.h_list[t]
            xh = np.concatenate([x, self.h_list[t-1] if t > 0 else np.zeros_like(h)])
            z = self.sigmoid(np.dot(xh, self.W_z) + self.b_z)
            r = self.sigmoid(np.dot(xh, self.W_r) + self.b_r)
            rh = r * (self.h_list[t-1] if t > 0 else np.zeros_like(h))
            xrh = np.concatenate([x, rh])
            h_hat = np.tanh(np.dot(xrh, self.W_h) + self.b_h)

            dh = dhs + dh_next
            dz = dh * (h_hat - self.h_list[t-1] if t > 0 else 0)
            dh_hat = dh * z
            dh_hat_raw = (1 - h_hat ** 2) * dh_hat
            dxrh = np.dot(dh_hat_raw, self.W_h.T)
            dW_h = np.outer(xrh, dh_hat_raw)
            self.dW_h += dW_h
            db_h += dh_hat_raw

            drh = dxrh[len(x):]
            dr = drh * (self.h_list[t-1] if t > 0 else 0)
            dr_raw = dr * r * (1 - r)
            dxh_r = np.outer(xh, dr_raw)
            self.dW_r += dxh_r
            db_r += dr_raw

            dz_raw = dz * z * (1 - z)
            dxh_z = np.outer(xh, dz_raw)
            self.dW_z += dxh_z
            db_z += dz_raw

            dh_next = dh * (1 - z) + drh * r

        return dh_next

class GRUDecoder:
    def __init__(self, vocab_size, embed_size, hidden_size):
        self.embedding = np.random.randn(vocab_size, embed_size) * 0.01
        self.W_z = np.random.randn(embed_size + hidden_size, hidden_size) * 0.01
        self.W_r = np.random.randn(embed_size + hidden_size, hidden_size) * 0.01
        self.W_h = np.random.randn(embed_size + hidden_size, hidden_size) * 0.01
        self.W_ah = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, vocab_size) * 0.01
        self.b_z = np.zeros(hidden_size)
        self.b_r = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(vocab_size)

        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_v = np.random.randn(hidden_size, hidden_size) * 0.01

        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_h = np.zeros_like(self.W_h)
        self.dW_ah = np.zeros_like(self.W_ah)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def attention(self, h, enc_outputs):
        Q = np.dot(h, self.W_q)
        Ks = np.dot(enc_outputs, self.W_k.T)
        Vs = np.dot(enc_outputs, self.W_v.T)
        scores = np.dot(Ks, Q) / np.sqrt(Ks.shape[1])
        weights = self.softmax(scores)
        context = np.sum(weights[:, None] * Vs, axis=0)
        return context, weights

    def forward(self, enc_outputs, y_seq):
        self.outputs = []
        self.y_seq = y_seq
        self.h = enc_outputs[-1]
        loss = 0

        for t in range(len(y_seq) - 1):
            input_idx = y_seq[t]
            target_idx = y_seq[t + 1]
            x = self.embedding[input_idx]

            context, _ = self.attention(self.h, enc_outputs)
            xc = x + np.dot(context, self.W_ah)

            xh = np.concatenate([xc, self.h])
            z = self.sigmoid(np.dot(xh, self.W_z) + self.b_z)
            r = self.sigmoid(np.dot(xh, self.W_r) + self.b_r)
            rh = r * self.h
            xrh = np.concatenate([xc, rh])
            h_hat = np.tanh(np.dot(xrh, self.W_h) + self.b_h)
            self.h = (1 - z) * self.h + z * h_hat

            y_pred = np.dot(self.h, self.W_hy) + self.b_y
            probs = self.softmax(y_pred)
            loss += -np.log(probs[target_idx] + 1e-8)

            self.outputs.append((x, context, xc, xh, r, rh, xrh, z, h_hat, self.h.copy(), y_pred, target_idx))

        return loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self):
        self.dW_z.fill(0)
        self.dW_r.fill(0)
        self.dW_h.fill(0)
        self.dW_ah.fill(0)
        self.dW_hy.fill(0)
        self.dW_q.fill(0)
        self.dW_k.fill(0)
        self.dW_v.fill(0)
        db_z = np.zeros_like(self.b_z)
        db_r = np.zeros_like(self.b_r)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(self.h)

        for t in reversed(range(len(self.outputs))):
            x, context, xc, xh, r, rh, xrh, z, h_hat, h, y_pred, target_idx = self.outputs[t]
            dy = self.softmax(y_pred)
            dy[target_idx] -= 1
            self.dW_hy += np.outer(h, dy)
            db_y += dy
            dh = np.dot(dy, self.W_hy.T) + dh_next

            dz = dh * (h_hat - h)
            dh_hat = dh * z
            dh_hat_raw = (1 - h_hat ** 2) * dh_hat
            self.dW_h += np.outer(xrh, dh_hat_raw)
            db_h += dh_hat_raw
            dxrh = np.dot(dh_hat_raw, self.W_h.T)
            drh = dxrh[len(x):]
            dr = drh * h
            dr_raw = dr * r * (1 - r)
            self.dW_r += np.outer(xh, dr_raw)
            db_r += dr_raw
            dz_raw = dz * z * (1 - z)
            self.dW_z += np.outer(xh, dz_raw)
            db_z += dz_raw
            dh_next = dh * (1 - z) + drh * r

            # 어텐션 backward는 여기선 생략 (간단화)
            self.dW_ah += np.outer(context, xc - x)

        return dh_next

    def decode(self, encoder_outputs, word2idx, idx2word, max_len=25, temperature=1.0, top_k=5):
        h = encoder_outputs[-1]
        idx = word2idx["<start>"]
        output = []

        for _ in range(max_len):
            input_emb = self.embedding[idx]
            context, _ = self.attention(h, encoder_outputs)
            combined_input = input_emb + np.dot(context, self.W_ah)
            h = np.tanh(np.dot(combined_input, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            y_pred = np.dot(h, self.W_hy) + self.b_y
            y_pred /= temperature

            if top_k > 0:
                top_k_idx = np.argsort(y_pred)[-top_k:]
                top_k_probs = np.exp(y_pred[top_k_idx])
                top_k_probs /= np.sum(top_k_probs)
                idx = np.random.choice(top_k_idx, p=top_k_probs)
            else:
                probs = self.softmax(y_pred)
                idx = np.random.choice(len(probs), p=probs)

            word = idx2word.get(idx, "<unk>")
            if word == "<end>":
                break
            output.append(word)

        return " ".join(output)

class Seq2Seq:
    def __init__(self, encoder, decoder, learning_rate=0.001):
        self.encoder = encoder
        self.decoder = decoder
        self.lr = learning_rate

        self.optimizer = Adam(
        params = [
    encoder.W_z, encoder.W_r, encoder.W_h,
    encoder.b_z, encoder.b_r, encoder.b_h,

                # Decoder GRU parameters
                decoder.W_zx, decoder.W_zh, decoder.W_rx, decoder.W_rh, decoder.W_hx, decoder.W_hh,
                # Attention parameters
                decoder.W_a, decoder.v_a,
                # Output layer
                decoder.W_hy
            ],
            grads=[None] * 15,
            lr=self.lr
        )

    def train_step(self, x_seq, y_seq):
        enc_outputs = self.encoder.forward(x_seq)
        loss = self.decoder.forward(enc_outputs, y_seq)
        return loss

    def update_params(self):
        dencoder_h = self.decoder.backward()
        self.encoder.backward(dencoder_h)

        self.optimizer.grads = [
            # Encoder GRU gradients
            self.encoder.dW_zx, self.encoder.dW_zh,
            self.encoder.dW_rx, self.encoder.dW_rh,
            self.encoder.dW_hx, self.encoder.dW_hh,
            # Decoder GRU gradients
            self.decoder.dW_zx, self.decoder.dW_zh,
            self.decoder.dW_rx, self.decoder.dW_rh,
            self.decoder.dW_hx, self.decoder.dW_hh,
            # Attention gradients
            self.decoder.dW_a, self.decoder.dv_a,
            # Output layer gradients
            self.decoder.dW_hy
        ]

        self.optimizer.update()

    def predict(self, x_seq, word2idx, idx2word):
        enc_outputs = self.encoder.forward(x_seq)
        return self.decoder.decode(enc_outputs, word2idx, idx2word)