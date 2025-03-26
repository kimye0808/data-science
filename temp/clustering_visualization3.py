import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# 데이터 로드
df = pd.read_csv("./dataset/twcs/customer_support_twitter.csv")

# 고객 문의만 필터링 (inbound == True)
df_inbound = df[df['inbound'] == True]

# 500개 샘플링
sample_size = 500
df_sample = df_inbound.sample(n=sample_size, random_state=42)

# 텍스트 데이터 추출
texts = df_sample['text'].tolist()

# 영어 텍스트 전처리
def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = str(text).lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        processed_texts.append(text)
    return processed_texts

texts = preprocess_text(texts)

# BERT로 텍스트 임베딩
class BERTEmbedding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts, batch_size=32):
        embeddings = []
        full_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                for j in range(len(batch_texts)):
                    embeddings.append((batch_texts[j], batch_embeddings[j]))
                    full_embeddings.append(batch_embeddings[j])
        return embeddings, np.array(full_embeddings)

# 임베딩 생성
bert = BERTEmbedding()
embeddings, full_embeddings = bert.get_embeddings(texts)

# DBSCAN (NumPy 벡터화 적용, full_embeddings 사용)
def eucDistanceFunc(a, b):
    return cosine_similarity([a[1]], [b[1]])[0][0]

def rangeQuery(db, p, eps):
    db_vectors = np.array([item[1] for item in db])
    p_vector = db[p][1]
    similarities = cosine_similarity([p_vector], db_vectors)[0]
    neighbors = np.where(similarities >= eps)[0].tolist()
    return neighbors

def dbscan(db, eps, minPts):
    labels = len(db) * [None]
    c = 0

    for p in tqdm(range(len(db)), desc="DBSCAN"):
        if labels[p] is not None:
            continue

        neighborsN = rangeQuery(db, p, eps)
        if len(neighborsN) < minPts:
            labels[p] = -1
            continue

        c += 1
        labels[p] = c

        seedSet = set(neighborsN) - {p}
        while seedSet:
            q = seedSet.pop()
            if labels[q] == -1:
                labels[q] = c
            if labels[q] is not None:
                continue

            labels[q] = c
            neighborsQ = rangeQuery(db, q, eps)
            if len(neighborsQ) < minPts:
                continue
            seedSet.update(set(neighborsQ))

    return labels

# DBSCAN 실행
eps = 0.7
minPts = 5
labels = dbscan(embeddings, eps, minPts)

# 클러스터별 유형 식별
def identify_cluster_types(texts, labels, embeddings):
    cluster_texts = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append(embeddings[i][0])

    cluster_types = {}
    for cid, texts in cluster_texts.items():
        text_combined = " ".join(texts)
        if "order" in text_combined or "delivery" in text_combined:
            cluster_types[cid] = "Delivery Inquiry"
        elif "payment" in text_combined or "bill" in text_combined:
            cluster_types[cid] = "Payment Issue"
        elif "service" in text_combined or "support" in text_combined:
            cluster_types[cid] = "Service Issue"
        else:
            cluster_types[cid] = "Other"
    return cluster_types

cluster_types = identify_cluster_types(texts, labels, embeddings)

# Silhouette Score 계산 (Noise 제외)
valid_indices = [i for i, label in enumerate(labels) if label != -1]
valid_embeddings = full_embeddings[valid_indices]
valid_labels = [labels[i] for i in valid_indices]

if len(set(valid_labels)) > 1:
    silhouette_avg = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
    print(f"Silhouette Score: {silhouette_avg:.3f}")
else:
    print("Silhouette Score 계산 불가: 클러스터가 1개 또는 Noise만 존재")

# 데이터프레임 생성 (response 제외)
plot_data = pd.DataFrame({
    'text': texts,
    'cluster': [str(label) for label in labels],
    'type': [cluster_types.get(label, "Noise") if label != -1 else "Noise" for label in labels]
})

# 시각화
# TSNE로 2차원 축소
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
embeddings_2d = tsne.fit_transform(full_embeddings)

plot_data['x'] = embeddings_2d[:, 0]
plot_data['y'] = embeddings_2d[:, 1]

# 색상 맵 정의
color_discrete_map = {
    'Noise': 'gray',
    'Delivery Inquiry': 'blue',
    'Payment Issue': 'red',
    'Service Issue': 'green',
    'Other': 'purple'
}

# Plotly Express로 2D 산점도 생성
fig_2d = px.scatter(
    plot_data,
    x='x',
    y='y',
    color='type',
    hover_data=['text', 'type'],
    title="Interactive DBSCAN Clustering of Customer Inquiries (2D)",
    labels={'x': 'TSNE Component 1', 'y': 'TSNE Component 2'},
    color_discrete_map=color_discrete_map
)

# 레이아웃 조정
fig_2d.update_layout(
    width=800,
    height=600,
    showlegend=True,
    title_x=0.5,
    legend_title_text='Cluster Type'
)

# 2D 그래프 표시
fig_2d.show()

# 2D HTML 파일로 저장
fig_2d.write_html("clustering_result3_2d.html")

# 3D 시각화
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=300)
embeddings_3d = tsne_3d.fit_transform(full_embeddings)
plot_data['z'] = embeddings_3d[:, 2]

# Plotly Express로 3D 산점도 생성
fig_3d = px.scatter_3d(
    plot_data,
    x='x',
    y='y',
    z='z',
    color='type',
    hover_data=['text', 'type'],
    title="Interactive DBSCAN Clustering of Customer Inquiries (3D)",
    labels={'x': 'TSNE Component 1', 'y': 'TSNE Component 2', 'z': 'TSNE Component 3'},
    color_discrete_map=color_discrete_map
)

# 레이아웃 조정
fig_3d.update_layout(
    width=800,
    height=600,
    showlegend=True,
    title_x=0.5,
    legend_title_text='Cluster Type'
)

# 3D 그래프 표시
fig_3d.show()

# 3D HTML 파일로 저장
fig_3d.write_html("clustering_result3_3d.html")