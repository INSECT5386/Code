# LiteSeq_Attn.py    
    
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
            if grad is None:    
                continue    
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad    
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)    
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)    
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)    
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)    
    
    
  
  
class SimpleEncoder:  
    def __init__(self, vocab_size, embed_size, hidden_size):  
        self.embedding = np.random.randn(vocab_size, embed_size) * 0.01  
        self.W_xh = np.random.randn(embed_size, hidden_size) * 0.01  
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  
        self.b_h = np.zeros(hidden_size)  
  
        self.dW_xh = np.zeros_like(self.W_xh)  
        self.dW_hh = np.zeros_like(self.W_hh)  
  
    def forward(self, x):  
        self.h = np.zeros_like(self.b_h)  
        self.h_list = []  
  
        # None 방지 필터  
        x = [idx for idx in x if idx is not None and isinstance(idx, int)]  
  
        self.x_embeds = [self.embedding[idx] for idx in x]  
  
        for x_emb in self.x_embeds:  
            self.h = np.tanh(np.dot(x_emb, self.W_xh) + np.dot(self.h, self.W_hh) + self.b_h)  
            self.h_list.append(self.h.copy())  
  
        # 빈 입력에 대한 안전장치  
        if len(self.h_list) == 0:  
            h_zero = np.zeros_like(self.b_h)  
            self.h_list.append(h_zero)  
  
        return np.stack(self.h_list)  
  
    def backward(self, dh):  
        self.dW_xh.fill(0)  
        self.dW_hh.fill(0)  
        db_h = np.zeros_like(self.b_h)  
        dh_t = dh  
  
        for t in reversed(range(len(self.x_embeds))):  
            h_t = self.h_list[t]  
            x_emb = self.x_embeds[t]  
            dtanh = (1 - h_t ** 2) * dh_t  
            self.dW_xh += np.outer(x_emb, dtanh)  
            self.dW_hh += np.outer(h_t, dtanh)  
            db_h += dtanh  
            dh_t = np.dot(dtanh, self.W_hh.T)  
  
        return dh_t  
  
import numpy as np  
  
class SimpleDecoder:  
    def __init__(self, vocab_size, embed_size, hidden_size):  
        self.embedding = np.random.randn(vocab_size, embed_size) * 0.01  
        self.W_xh = np.random.randn(embed_size, hidden_size) * 0.01  
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  
        self.W_ah = np.random.randn(hidden_size, hidden_size) * 0.01  
        self.W_hy = np.random.randn(hidden_size, vocab_size) * 0.01  
  
        # 어텐션: dot-product 기반  
        self.W_q = np.random.randn(hidden_size, hidden_size) * 0.01  
        self.W_k = np.random.randn(hidden_size, hidden_size) * 0.01  
        self.W_v = np.random.randn(hidden_size, hidden_size) * 0.01  
  
        self.b_h = np.zeros(hidden_size)  
        self.b_y = np.zeros(vocab_size)  
  
        self.dW_xh = np.zeros_like(self.W_xh)  
        self.dW_hh = np.zeros_like(self.W_hh)  
        self.dW_ah = np.zeros_like(self.W_ah)  
        self.dW_hy = np.zeros_like(self.W_hy)  
        self.dW_q = np.zeros_like(self.W_q)  
        self.dW_k = np.zeros_like(self.W_k)  
        self.dW_v = np.zeros_like(self.W_v)  
  
    def softmax(self, x):  
        exp_x = np.exp(x - np.max(x))  
        return exp_x / np.sum(exp_x)  
  
    def attention(self, decoder_hidden, encoder_outputs):  
        Q = np.dot(decoder_hidden, self.W_q)  # (H,)  
        Ks = np.dot(encoder_outputs, self.W_k.T)  # (T, H)  
        Vs = np.dot(encoder_outputs, self.W_v.T)  # (T, H)  
  
        scores = np.dot(Ks, Q) / np.sqrt(Ks.shape[1])  # scaled dot-product  
        attn_weights = self.softmax(scores)  
        context = np.sum(attn_weights[:, None] * Vs, axis=0)  
        return context, attn_weights  
  
    def forward(self, encoder_outputs, y_seq):  
        self.outputs = []  
        self.y_seq = y_seq  
        self.hidden = encoder_outputs[-1]  
        loss = 0  
  
        for t in range(len(y_seq) - 1):  
            input_idx = y_seq[t]  
            target_idx = y_seq[t + 1]  
            input_emb = self.embedding[input_idx]  
  
            context, _ = self.attention(self.hidden, encoder_outputs)  
            combined_input = input_emb + np.dot(context, self.W_ah)  
  
            prev_hidden = self.hidden.copy()  
            self.hidden = np.tanh(np.dot(combined_input, self.W_xh) + np.dot(prev_hidden, self.W_hh) + self.b_h)  
            y_pred = np.dot(self.hidden, self.W_hy) + self.b_y  
  
            target_onehot = np.zeros_like(y_pred)  
            target_onehot[target_idx] = 1  
            y_pred_prob = self.softmax(y_pred)  
            loss += -np.sum(target_onehot * np.log(y_pred_prob + 1e-8))  
  
            self.outputs.append((y_pred, self.hidden.copy(), input_idx, context, combined_input, prev_hidden))  
  
        return loss  
  
    def backward(self):  
        self.dW_xh.fill(0)  
        self.dW_hh.fill(0)  
        self.dW_hy.fill(0)  
        self.dW_ah.fill(0)  
        self.dW_q.fill(0)  
        self.dW_k.fill(0)  
        self.dW_v.fill(0)  
        db_h = np.zeros_like(self.b_h)  
        db_y = np.zeros_like(self.b_y)  
        dh_next = np.zeros_like(self.hidden)  
  
        for t in reversed(range(len(self.outputs))):  
            y_pred, h_t, input_idx, context, combined_input, prev_h = self.outputs[t]  
            target_idx = self.y_seq[t + 1]  
  
            target_onehot = np.zeros_like(y_pred)  
            target_onehot[target_idx] = 1  
            dy = 2 * (y_pred - target_onehot)  
  
            self.dW_hy += np.outer(h_t, dy)  
            db_y += dy  
            dh = np.dot(dy, self.W_hy.T) + dh_next  
  
            dtanh = (1 - h_t ** 2) * dh  
            self.dW_xh += np.outer(combined_input, dtanh)  
            self.dW_hh += np.outer(prev_h, dtanh)  
            self.dW_ah += np.outer(context, dtanh)  
            db_h += dtanh  
  
            # 어텐션 역전파  
            # 1. 어텐션 가중치에 대한 기울기  
            attn_grad = np.dot(dy, self.W_hy.T)  # (H,)  
  
            # 2. 어텐션에서 사용하는 Q, K, V에 대한 기울기  
            dQ = np.dot(attn_grad, self.W_q.T)  # (H,)  
            dK = np.dot(attn_grad, self.W_k.T)  # (T, H)  
            dV = np.dot(attn_grad, self.W_v.T)  # (T, H)  
  
            # Q, K, V에 대한 기울기를 W_q, W_k, W_v에 적용  
            self.dW_q += np.outer(dQ, self.hidden)  
            self.dW_k += np.outer(dK, self.hidden)  
            self.dW_v += np.outer(dV, self.hidden)  
  
            dh_next = np.dot(dtanh, self.W_hh.T)  
  
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
            params=[    
                encoder.W_xh, encoder.W_hh,    
                decoder.W_xh, decoder.W_hh, decoder.W_hy, decoder.W_ah    
            ],    
            grads=[None] * 6,    
            lr=self.lr    
        )    
    
    def train_step(self, x_seq, y_seq):    
        enc_outputs = self.encoder.forward(x_seq)    
        loss = self.decoder.forward(enc_outputs, y_seq)    
        return loss    
    
    def update_params(self):      
        dh_decoder = self.decoder.backward()      
        self.encoder.backward(dh_decoder)      
        
    # 어텐션 가중치 기울기도 포함시켜야 한다.    
        self.optimizer.grads = [      
            self.encoder.dW_xh, self.encoder.dW_hh,      
            self.decoder.dW_xh, self.decoder.dW_hh,      
            self.decoder.dW_hy, self.decoder.dW_ah,      
            self.decoder.dW_q, self.decoder.dW_k, self.decoder.dW_v  # 어텐션 가중치 기울기 추가    
        ]      
        
        self.optimizer.update()
    
    def predict(self, x_seq, word2idx, idx2word):    
        enc_outputs = self.encoder.forward(x_seq)    
        return self.decoder.decode(enc_outputs, word2idx, idx2word)
