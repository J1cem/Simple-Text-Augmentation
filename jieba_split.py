import json
import jieba
from tqdm import tqdm


def preprocess_and_save(json_file, output_file):
    # 打开 JSON 文件并读取数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 使用 tqdm 来可视化进度条
        for item in tqdm(data, desc="Processing items", ncols=100, unit="item"):
            # 对 title 和 content 使用 jieba 分词
            title = " ".join(jieba.cut(item['title']))  # 分词后将词语用空格连接
            content = " ".join(jieba.cut(item['content']))

            # 将分词后的结果写入文件，标题和内容分开
            out_f.write(f"Title: {title}\n")
            out_f.write(f"Content: {content}\n\n")


# 示例调用
json_file = 'new2016zh_data_2.json'  # 输入你的 JSON 文件路径
output_file = 'processed_dataset-3.txt'  # 输出的文本文件路径
preprocess_and_save(json_file, output_file)