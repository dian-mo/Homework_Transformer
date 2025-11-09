# 导入需要的库
import torch
import torch.nn as nn
import math
import numpy as np
import os
import jieba
import nltk
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
try:
    # 尝试使用系统中存在的中文字体
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')  # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    try:
        font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # Mac
        plt.rcParams['font.sans-serif'] = ['STHeiti']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：未找到中文字体，图表可能无法正确显示中文")
        font = FontProperties()

# 下载nltk分词所需资源
nltk.download('punkt')


# ---------------------- 数据预处理函数 ----------------------
def denoise_line(line):
    """处理包含##的文本，将##开头的token与前一个token合并"""
    tokens = line.split()
    result = []
    for token in tokens:
        if token.startswith('##'):
            if result:
                result[-1] += token[2:]
            else:
                result.append(token[2:])
        else:
            result.append(token)
    return ' '.join(result)


# ---------------------- 可视化模块 ----------------------
class TrainingVisualizer:
    """训练过程可视化类"""

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.test_ppls = []
        self.epochs = []

    def update(self, epoch, train_loss, test_loss, test_ppl):
        """更新训练数据"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_ppls.append(test_ppl)

    def plot_training_curves(self, save_path='training_curves.png'):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(self.epochs, self.test_losses, 'r-', label='测试损失', linewidth=2)
        ax1.set_xlabel('训练轮次', fontproperties=font)
        ax1.set_ylabel('损失值', fontproperties=font)
        ax1.set_title('训练和测试损失曲线', fontproperties=font, fontsize=14)
        ax1.legend(prop=font)
        ax1.grid(True, alpha=0.3)

        # 困惑度曲线
        ax2.plot(self.epochs, self.test_ppls, 'g-', label='测试困惑度', linewidth=2)
        ax2.set_xlabel('训练轮次', fontproperties=font)
        ax2.set_ylabel('困惑度', fontproperties=font)
        ax2.set_title('测试困惑度曲线', fontproperties=font, fontsize=14)
        ax2.legend(prop=font)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_attention_heatmap(self, attention_weights, src_tokens, tgt_tokens,
                               save_path='attention_heatmap.png'):
        """绘制注意力热力图"""
        if len(attention_weights.shape) == 4:
            # 取第一个样本，第一个头的注意力权重
            attn = attention_weights[0, 0].cpu().numpy()
        else:
            attn = attention_weights[0].cpu().numpy()

        plt.figure(figsize=(12, 8))
        sns.heatmap(attn,
                    xticklabels=src_tokens[:attn.shape[1]],
                    yticklabels=tgt_tokens[:attn.shape[0]],
                    cmap='YlOrRd',
                    cbar_kws={'label': '注意力权重'})
        plt.xlabel('源序列', fontproperties=font)
        plt.ylabel('目标序列', fontproperties=font)
        plt.title('注意力机制热力图', fontproperties=font, fontsize=16)
        plt.xticks(rotation=45, fontproperties=font)
        plt.yticks(rotation=0, fontproperties=font)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_results_comparison(self, test_results, save_path='results_comparison.png'):
        """绘制结果对比图"""
        examples = list(range(1, len(test_results) + 1))
        input_lengths = [len(result['input_tokens']) for result in test_results]
        target_lengths = [len(result['target_tokens']) for result in test_results]
        pred_lengths = [len(result['pred_tokens']) for result in test_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 序列长度对比
        x = np.arange(len(examples))
        width = 0.25
        ax1.bar(x - width, input_lengths, width, label='输入序列', alpha=0.7)
        ax1.bar(x, target_lengths, width, label='真实归纳', alpha=0.7)
        ax1.bar(x + width, pred_lengths, width, label='模型归纳', alpha=0.7)
        ax1.set_xlabel('测试样例', fontproperties=font)
        ax1.set_ylabel('序列长度', fontproperties=font)
        ax1.set_title('序列长度对比', fontproperties=font, fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'样例 {i}' for i in examples])
        ax1.legend(prop=font)

        # 词汇重叠率
        overlap_rates = []
        for result in test_results:
            target_set = set(result['target_tokens'])
            pred_set = set(result['pred_tokens'])
            if len(target_set) > 0:
                overlap = len(target_set & pred_set) / len(target_set)
            else:
                overlap = 0
            overlap_rates.append(overlap)

        ax2.bar(examples, overlap_rates, alpha=0.7, color='orange')
        ax2.set_xlabel('测试样例', fontproperties=font)
        ax2.set_ylabel('词汇重叠率', fontproperties=font)
        ax2.set_title('真实归纳与模型归纳的词汇重叠率', fontproperties=font, fontsize=14)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_word_cloud(self, texts, title, save_path='wordcloud.png'):
        """生成词云图"""
        if not texts:
            return

        # 合并所有文本
        all_text = ' '.join([' '.join(text) for text in texts])

        # 生成词云
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path=r'C:\Windows\Fonts\simhei.ttf' if os.path.exists(r'C:\Windows\Fonts\simhei.ttf') else None
        ).generate(all_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontproperties=font, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ---------------------- 核心模块保持不变 ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须是num_heads的整数倍"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask)[0])
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x = self.sublayer1(x, lambda x: self.masked_self_attn(x, x, x, tgt_mask)[0])
        x = self.sublayer2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask)[0])
        x = self.sublayer3(x, self.ffn)
        return x


def create_mask(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2).bool()
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3).bool()
    tgt_len = tgt.size(1)
    tgt_future_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).unsqueeze(0).unsqueeze(1).bool()
    tgt_mask = tgt_pad_mask & tgt_future_mask
    return src_mask, tgt_mask


# ---------------------- 改进的词表和数据处理 ----------------------
class Vocab:
    """改进的词表类：降低词频阈值，增加词表大小"""

    def __init__(self, min_freq=1):  # 降低min_freq从2到1
        self.min_freq = min_freq
        self.idx2token = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.token2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.freq = {}

    def build_vocab(self, token_list):
        """构建词表，记录所有出现过的词汇"""
        for token in token_list:
            self.freq[token] = self.freq.get(token, 0) + 1
        # 包含所有出现过的词汇，不进行过滤
        for token, freq in self.freq.items():
            if freq >= self.min_freq:
                self.token2idx[token] = len(self.token2idx)
                self.idx2token.append(token)
        print(f"词表构建完成，共 {len(self.token2idx)} 个词汇")

    def encode(self, token_list, max_len=50):
        """编码序列，处理未知词"""
        idx_list = [self.token2idx['<BOS>']] + [self.token2idx.get(t, 1) for t in token_list] + [
            self.token2idx['<EOS>']]
        if len(idx_list) > max_len:
            idx_list = idx_list[:max_len]
        else:
            idx_list += [0] * (max_len - len(idx_list))
        return torch.tensor(idx_list, dtype=torch.long)

    def __len__(self):
        return len(self.token2idx)


def load_data(file_path):
    """加载数据并统计基本信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    processed_lines = []
    for line in lines:
        processed_line = denoise_line(line)
        processed_lines.append(processed_line)

    print(f"从 {file_path} 加载了 {len(processed_lines)} 条数据")
    return processed_lines


def tokenize(sent, is_english=True):
    """改进的分词函数，保留更多信息"""
    sent = sent.replace('##', '')
    if is_english:
        # 保留原始大小写，因为专有名词很重要
        tokens = nltk.word_tokenize(sent)
        return tokens
    else:
        return list(jieba.cut(sent))


class SummarizationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab, max_len=50):
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src_token = tokenize(self.src_sents[idx], is_english=True)
        tgt_token = tokenize(self.tgt_sents[idx], is_english=True)
        src_idx = self.src_vocab.encode(src_token, self.max_len)
        tgt_idx = self.tgt_vocab.encode(tgt_token, self.max_len)
        return src_idx, tgt_idx


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_pad, tgt_pad


class Transformer(nn.Module):
    """增大模型容量"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_len=50,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoders = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_mask, tgt_mask = create_mask(src, tgt)
        src_emb = self.src_emb(src) * math.sqrt(self.d_model)
        src_pos = self.pos_enc(src_emb)
        enc_out = src_pos
        for enc in self.encoders:
            enc_out = enc(enc_out, src_mask)
        tgt_emb = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        tgt_pos = self.pos_enc(tgt_emb)
        dec_out = tgt_pos
        for dec in self.decoders:
            dec_out = dec(dec_out, enc_out, tgt_mask, src_mask)
        return self.proj(dec_out)


# ---------------------- 改进的训练和评估逻辑 ----------------------
def train_step(model, loader, criterion, optimizer, device):
    """训练步骤，添加梯度裁剪和学习率调度"""
    model.train()
    total_loss = 0.0
    total_batch = len(loader)

    for batch_idx, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        # 更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item() * src.size(0)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batch:
            progress = (batch_idx + 1) / total_batch * 100
            avg_batch_loss = loss.item()
            print(
                f"  批次 [{batch_idx + 1}/{total_batch}] | 进度: {progress:.1f}% | 当前批次Loss: {avg_batch_loss:.4f}")

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def test_step(model, loader, criterion, device):
    """测试步骤，添加更多评估指标"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
            total_loss += loss.item() * src.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


def beam_search_decode(model, src, src_vocab, tgt_vocab, max_len=50, beam_size=3, device='cpu'):
    """使用束搜索改进解码"""
    model.eval()
    # 编码源序列
    src_tokens = tokenize(src, is_english=True)
    src_idx = src_vocab.encode(src_tokens, max_len).unsqueeze(0).to(device)

    # 初始化束
    start_token = tgt_vocab.token2idx['<BOS>']
    beams = [([start_token], 0.0)]  # (序列, 对数概率)

    for _ in range(max_len - 1):
        all_candidates = []
        for seq, score in beams:
            # 如果序列以EOS结束，直接保留
            if seq[-1] == tgt_vocab.token2idx['<EOS>']:
                all_candidates.append((seq, score))
                continue

            # 准备解码器输入
            tgt_idx = torch.tensor([seq], device=device)
            with torch.no_grad():
                logits = model(src_idx, tgt_idx)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_size, dim=-1)

            # 扩展候选序列
            for i in range(beam_size):
                next_token = topk_indices[0, i].item()
                next_score = score + torch.log(topk_probs[0, i]).item()
                new_seq = seq + [next_token]
                all_candidates.append((new_seq, next_score))

        # 选择top-k候选
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        # 如果所有序列都结束了，提前停止
        if all(seq[-1] == tgt_vocab.token2idx['<EOS>'] for seq, _ in beams):
            break

    # 返回最佳序列
    best_seq = beams[0][0]
    tokens = [tgt_vocab.idx2token[idx] for idx in best_seq if idx not in [0, 2, 3]]
    return ' '.join(tokens)


def summarize_sent(model, sent, src_vocab, tgt_vocab, max_len=50, device='cpu', use_beam_search=True):
    """改进的归纳函数，支持束搜索"""
    if use_beam_search:
        return beam_search_decode(model, sent, src_vocab, tgt_vocab, max_len, beam_size=3, device=device)
    else:
        # 回退到贪心解码
        model.eval()
        token = tokenize(sent, is_english=True)
        src_idx = src_vocab.encode(token, max_len).unsqueeze(0).to(device)
        tgt_idx = torch.tensor([[2]], device=device)
        for _ in range(max_len - 1):
            logits = model(src_idx, tgt_idx)
            next_idx = logits.argmax(dim=-1)[:, -1].unsqueeze(1)
            tgt_idx = torch.cat([tgt_idx, next_idx], dim=1)
            if next_idx.item() == 3:
                break
        tgt_token = [tgt_vocab.idx2token[idx] for idx in tgt_idx[0].cpu().numpy() if idx not in [0, 2, 3]]
        return ' '.join(tgt_token)


# ---------------------- 主函数：改进的训练配置 ----------------------
if __name__ == "__main__":
    nltk.download('punkt_tab')
    nltk.download('punkt')

    # 改进的训练参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '.'
    batch_size = 32  # 减小批次大小以适应更大的模型
    max_len = 60  # 增加序列长度
    d_model = 128  # 增大模型维度
    num_heads = 4  # 增加注意力头数
    num_layers = 2  # 增加层数
    d_ff = 512  # 增大前馈网络维度
    dropout = 0.2  # 增加dropout防止过拟合
    lr = 1e-4  # 降低学习率
    epochs = 5  # 增加训练轮数
    save_path = 'improved_model.pth'

    # 初始化可视化器
    visualizer = TrainingVisualizer()

    print("加载数据...")
    train_src = load_data(os.path.join(data_dir, 'train.src.10k'))
    train_tgt = load_data(os.path.join(data_dir, 'train.tgt.10k'))
    test_src = load_data(os.path.join(data_dir, 'test.src'))
    test_tgt = load_data(os.path.join(data_dir, 'test.tgt'))

    # 构建改进的词表
    print("构建词表...")
    src_vocab = Vocab(min_freq=1)  # 包含所有出现过的词汇
    tgt_vocab = Vocab(min_freq=1)

    all_src_token = []
    for sent in train_src:
        all_src_token.extend(tokenize(sent, is_english=True))
    all_tgt_token = []
    for sent in train_tgt:
        all_tgt_token.extend(tokenize(sent, is_english=True))

    src_vocab.build_vocab(all_src_token)
    tgt_vocab.build_vocab(all_tgt_token)

    print(f"输入词表大小：{len(src_vocab)}")
    print(f"输出词表大小：{len(tgt_vocab)}")

    # 统计UNK比例
    src_unk_count = sum(1 for token in all_src_token if token not in src_vocab.token2idx)
    tgt_unk_count = sum(1 for token in all_tgt_token if token not in tgt_vocab.token2idx)
    print(f"源语言UNK比例: {src_unk_count / len(all_src_token) * 100:.2f}%")
    print(f"目标语言UNK比例: {tgt_unk_count / len(all_tgt_token) * 100:.2f}%")

    # 构建DataLoader
    train_dataset = SummarizationDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    test_dataset = SummarizationDataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # 初始化改进的模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # 添加标签平滑
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_ppl = float('inf')
    print(f"\n开始训练（设备：{device}）...")
    for epoch in range(epochs):
        print(f"\n=== Epoch [{epoch + 1}/{epochs}] ===")
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        test_loss, test_ppl = test_step(model, test_loader, criterion, device)
        scheduler.step()

        # 更新可视化数据
        visualizer.update(epoch + 1, train_loss, test_loss, test_ppl)

        print(f"=== Epoch [{epoch + 1}/{epochs}] 结果 ===")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")

        if test_ppl < best_ppl:
            best_ppl = test_ppl
            torch.save(model.state_dict(), save_path)
            print(f"✅ 最佳模型已保存（PPL: {best_ppl:.2f}）")

    # 绘制训练曲线
    print("\n=== 生成训练可视化图表 ===")
    visualizer.plot_training_curves()

    # 测试改进的模型
    print("\n=== 改进后的归纳测试 ===")
    model.load_state_dict(torch.load(save_path))

    test_indices = min(5, len(test_src))
    test_results = []

    for i in range(test_indices):
        src_sent = test_src[i]
        true_summary = test_tgt[i]
        pred_summary = summarize_sent(model, src_sent, src_vocab, tgt_vocab, max_len, device, use_beam_search=True)

        print(f"例子 {i + 1}:")
        print(f"输入: {src_sent}")
        print(f"真实归纳: {true_summary}")
        print(f"模型归纳: {pred_summary}")
        print("-" * 80)

        # 收集结果用于可视化
        test_results.append({
            'input_tokens': tokenize(src_sent, is_english=True),
            'target_tokens': tokenize(true_summary, is_english=True),
            'pred_tokens': tokenize(pred_summary, is_english=True) if pred_summary else []
        })

    # 生成结果对比图
    if test_results:
        visualizer.plot_results_comparison(test_results)

        # 生成词云
        visualizer.generate_word_cloud(
            [result['target_tokens'] for result in test_results],
            '真实归纳词云'
        )
        visualizer.generate_word_cloud(
            [result['pred_tokens'] for result in test_results if result['pred_tokens']],
            '模型归纳词云'
        )

    print("\n✅ 所有可视化图表已生成完成！")