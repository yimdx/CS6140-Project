from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 选择一个已有的预训练模型
#    英文示例: "sentence-transformers/all-MiniLM-L6-v2"
#    中文示例: "uer/sbert-base-chinese-nli"  (或其他中文SBERT模型)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# 2. 准备一些测试句子
sentences = [
    "I love natural language processing.",
    "The weather is nice today.",
    "I enjoy working with text data."
]

# 3. 编码句子，得到句向量
sentence_embeddings = model.encode(sentences)

# 4. 查看结果
for idx, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Vector shape: {sentence_embeddings[idx].shape}")
    # print(f"Vector (first 10 dims): {sentence_embeddings[idx][:10]}\n")
