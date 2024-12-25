# -*- coding: utf-8 -*-
import logging
import sys
import gensim.models as word2vec
from gensim.models.word2vec import LineSentence, logger
import os
from tqdm import tqdm  # 引入tqdm库


def train_word2vec(dataset_paths, out_vector):
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # 将多个文件合并为一个输入文件，并显示进度条
    combined_data = []
    for dataset_path in tqdm(dataset_paths, desc="Processing Files", unit="file"):
        if os.path.exists(dataset_path):
            logger.info(f"Processing file: {dataset_path}")
            with open(dataset_path, 'r', encoding='utf-8') as file:
                combined_data.extend(file.readlines())
        else:
            logger.warning(f"File {dataset_path} does not exist.")

    # 将合并的文本写入一个临时文件
    temp_file = "temp_combined.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.writelines(combined_data)

    # 把语料变成句子集合
    sentences = LineSentence(temp_file)

    # 训练word2vec模型，使用tqdm来显示训练进度
    logger.info("Training Word2Vec model...")
    model = word2vec.Word2Vec(sentences, vector_size=64, sg=1, window=5, min_count=50, workers=5, epochs=3)

    # 保存word2vec模型
    model.save("word2vec_new.model")
    model.wv.save_word2vec_format(out_vector, binary=False)

    # 删除临时文件
    os.remove(temp_file)
    logger.info("Model training completed and saved.")


# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语的相似词
def calculate_most_similar(model, word):
    similar_words = model.wv.most_similar(word, topn=10)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


if __name__ == '__main__':
    # 这里传入一个文本文件路径的列表
    dataset_paths = ["processed_dataset.txt", "processed_dataset-2.txt", "processed_dataset-3.txt"]  # 示例文本文件路径
    out_vector = 'imp_.vector'

    # 训练word2vec模型并保存
    train_word2vec(dataset_paths, out_vector)
