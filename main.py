import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import json
import jieba  # 导入jieba库

# 假设你已经训练好了Word2Vec模型
# 这里加载一个预训练的Word2Vec模型，或者你也可以自己训练一个模型
w2v_model = Word2Vec.load("word2vec_new.model")

# 计算余弦相似度
def compute_similarity(word_vector, context_vector):
    return cosine_similarity([word_vector], [context_vector])[0][0]

# 使用jieba分词
def cut_sentence(sentence):
    """
    使用jieba进行中文句子的分词
    """
    return list(jieba.cut(sentence))


class SentenceDataset(Dataset):
    def __init__(self, sentences, word2vec_model):
        self.sentences = sentences
        self.w2v_model = word2vec_model

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_vectors = []
        valid_words = []  # 记录有效词的标记

        # 获取每个词的词向量
        for word in sentence:
            if word in self.w2v_model.wv:
                word_vectors.append(self.w2v_model.wv[word])
                valid_words.append(True)
            else:
                word_vectors.append(np.zeros(self.w2v_model.vector_size))  # 若词不在词表中，使用零向量
                valid_words.append(False)  # 不在词表中的词标记为无效词

        # 将列表转换为NumPy数组再转换为Tensor
        word_vectors = np.array(word_vectors)
        word_vectors = torch.tensor(word_vectors).float()  # 确保是float类型

        return word_vectors, valid_words  # 返回有效词标记


class AttentionNet(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionNet, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(embedding_dim))  # 学习得到的注意力权重
        self.embedding_dim = embedding_dim

    def forward(self, word_vectors):
        """
        :param word_vectors: 词向量组成的列表 [batch_size, seq_len, embedding_dim]
        """
        # 计算每个词的注意力
        attention_scores = torch.matmul(word_vectors, self.attention_weights)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return attention_weights


# 准备训练数据
def load_sentences_from_file(file_path):
    """
    从文本文件中读取句子，每行是一个句子，单词用空格分割
    :param file_path: 文本文件路径
    :return: 句子列表，每个句子是一个单词列表
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行末的换行符并按空格分割单词
            sentence = line.strip().split()  # 使用strip去掉换行符，split按空格分割单词
            sentences.append(sentence)
    return sentences

# 示例：读取文件并打印结果
file_path = 'processed_dataset.txt'  # 请根据您的文件路径修改
sentences = load_sentences_from_file(file_path)

# 创建数据集和DataLoader
dataset = SentenceDataset(sentences, w2v_model)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义模型和优化器
embedding_dim = 64  # 假设Word2Vec的向量维度是100
model = AttentionNet(embedding_dim)
optimizer = Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    for word_vectors, valid_words in dataloader:
        word_vectors = word_vectors.squeeze(0)  # 去除batch维度
        word_vectors = word_vectors.float()  # 确保是float类型
        attention_weights = model(word_vectors)  # 计算注意力权重
        print(attention_weights)  # 查看每个词的注意力权重

        # 假设我们有某种损失函数和标签，这里只是展示模型训练流程
        loss = torch.mean(attention_weights)  # 这里只是一个示范，实际应该有更合理的损失函数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



# 步骤1：计算实际的注意力权重
def compute_attention_weights_for_sentence(sentence, model, w2v_model):
    """
    计算给定句子的每个词的注意力权重
    """
    word_vectors = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]  # 只选取有效词
    word_vectors = torch.tensor(word_vectors)  # 转换为tensor
    word_vectors = word_vectors.float()  # 确保是float类型
    attention_weights = model(word_vectors)  # 通过模型计算注意力权重
    return attention_weights.detach().numpy()  # 转为NumPy数组返回，方便后续处理


# 步骤2：替换低注意力词
def replace_low_attention_words(sentence, attention_weights, valid_words, threshold=0.2, max_sim_words=5,
                                max_similarity=0.8):
    """
    替换低注意力词，同时遵循最大相似度和最大替换词数量限制
    :param sentence: 句子
    :param attention_weights: 注意力权重
    :param valid_words: 有效词标记
    :param threshold: 注意力权重阈值，低于该值的词会被替换
    :param max_sim_words: 最大替换词数量
    :param max_similarity: 最大相似度，替换词必须大于此相似度
    :return: 新的句子
    """
    new_sentence = []

    # 确保 attention_weights 是一个张量或列表
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().numpy()  # 如果是tensor，转换为numpy数组

    # 使用模型中第一个有效词的维度初始化上下文向量
    try:
        context_vector = np.zeros_like(w2v_model.wv[sentence[0]])  # 使用第一个有效词的词向量维度进行初始化
    except KeyError:
        # 如果第一个词不在词表中，则跳过
        context_vector = np.zeros(w2v_model.vector_size)  # 默认为 Word2Vec 的词向量维度

    # 确保使用 64 维的向量维度
    if len(context_vector) != 64:
        context_vector = np.zeros(64)  # 重新初始化为 64 维向量

    # 计算上下文向量（加权平均）
    for word, alpha, valid in zip(sentence, attention_weights, valid_words):
        if not valid:
            continue  # 跳过无效词
        try:
            word_vector = w2v_model.wv[word]
            if len(word_vector) == 64:  # 确保 word_vector 也是 64 维
                context_vector += alpha * word_vector
        except KeyError:
            # 如果词不在词表中，跳过该词
            continue

    # 生成新句子，替换低注意力词
    for idx, (word, alpha, valid) in enumerate(zip(sentence, attention_weights, valid_words)):
        if not valid:
            new_sentence.append(word)
            continue

        if alpha < threshold:  # 注意力小于阈值，进行替换
            try:
                similar_words = w2v_model.wv.most_similar(word, topn=max_sim_words)
                replacement_word = None
                for similar_word, similarity in similar_words:
                    if similarity >= max_similarity:  # 满足最大相似度
                        replacement_word = similar_word
                        break

                if replacement_word:
                    new_sentence.append(replacement_word)
                else:
                    new_sentence.append(word)
            except KeyError:
                # 如果某个词不在词表中，跳过
                new_sentence.append(word)
        else:
            new_sentence.append(word)

    return new_sentence


def enhance_text_from_file(input_file, output_file, model, w2v_model, threshold=0.2, max_sim_words=5,
                           max_similarity=0.8):
    """
    从文件读取句子，增强每个句子，并保存原始句子和增强后的句子到JSON文件中
    :param input_file: 输入的txt文件路径
    :param output_file: 输出的json文件路径
    :param model: 训练好的注意力模型
    :param w2v_model: 训练好的Word2Vec模型
    :param threshold: 注意力权重的阈值，低于该值的词会被替换
    :param max_sim_words: 最大替换词数量
    :param max_similarity: 替换词的最小相似度
    """
    enhanced_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()  # 去掉换行符

            # 处理句子并获得增强后的句子
            enhanced_sentence = process_sentence(sentence, model, w2v_model, threshold, max_sim_words, max_similarity)

            # 将增强后的句子拼接成一整句话，不添加空格
            enhanced_sentence_str = ''.join(enhanced_sentence)  # 直接连接所有词语

            # 保存原始句子和增强后的句子
            enhanced_data.append({
                'original': sentence,
                'enhanced': enhanced_sentence_str  # 增强后的句子
            })

    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(enhanced_data, json_file, ensure_ascii=False, indent=4)


def process_sentence(sentence, model, w2v_model, threshold=0.2, max_sim_words=5, max_similarity=0.8):
    """
    处理给定的中文句子，包括分词、计算注意力权重、替换低注意力词
    :param sentence: 输入的中文句子
    :param model: 训练好的注意力模型
    :param w2v_model: 训练好的Word2Vec模型
    :param threshold: 注意力权重的阈值，低于该值的词会被替换
    :param max_sim_words: 最大替换词数量
    :param max_similarity: 替换词的最小相似度
    :return: 新的句子（可能替换了低注意力词）
    """
    # 步骤1：使用jieba分词
    segmented_sentence = cut_sentence(sentence)

    # 计算每个词的注意力权重
    attention_weights = compute_attention_weights_for_sentence(segmented_sentence, model, w2v_model)

    # 步骤2：替换低注意力词
    valid_words = [True if word in w2v_model.wv else False for word in segmented_sentence]
    new_sentence = replace_low_attention_words(segmented_sentence, attention_weights, valid_words,
                                               threshold, max_sim_words, max_similarity)

    return new_sentence


input_file = '1234.txt'  # 输入的txt文件路径
output_file = 'enhanced_data.json'    # 输出的json文件路径

enhance_text_from_file(input_file, output_file, model, w2v_model)
print(f"增强数据已保存到 {output_file}")



