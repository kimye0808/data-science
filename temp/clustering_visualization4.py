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
import time
import pickle
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# 데이터 로드
start_time = time.time()
df = pd.read_csv("./dataset/twcs/customer_support_twitter.csv")
print(f"데이터 로드 시간: {time.time() - start_time:.2f}초")

# 고객 문의만 필터링 (inbound == True)
start_time = time.time()
df_inbound = df[df['inbound'] == True]
print(f"고객 문의 필터링 시간: {time.time() - start_time:.2f}초")

# 단순 무작위 샘플링
start_time = time.time()
df_sample = df_inbound.sample(n=500, random_state=42)
print(f"샘플링된 데이터 크기: {len(df_sample)}")
print(f"샘플링 시간: {time.time() - start_time:.2f}초")

# 텍스트 및 브랜드 데이터 추출
texts = df_sample['text'].tolist()
authors = df_sample['author_id'].tolist()

# 영어 텍스트 전처리
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = str(text).lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        tokens = word_tokenize(text)
        tokens = [ps.stem(token) for token in tokens if token not in stop_words]
        processed_texts.append(' '.join(tokens))
    return processed_texts

start_time = time.time()
texts = preprocess_text(texts)
print(f"텍스트 전처리 시간: {time.time() - start_time:.2f}초")

# BERT로 텍스트 임베딩
class BERTEmbedding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = BertModel.from_pretrained("distilbert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts, batch_size=16):
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

# 임베딩 생성 (캐싱 사용)
embedding_cache_file = "embeddings.pkl"
if os.path.exists(embedding_cache_file):
    with open(embedding_cache_file, 'rb') as f:
        embeddings, full_embeddings = pickle.load(f)
    print("임베딩 로드 완료 (캐시 사용)")
else:
    start_time = time.time()
    bert = BERTEmbedding()
    embeddings, full_embeddings = bert.get_embeddings(texts)
    with open(embedding_cache_file, 'wb') as f:
        pickle.dump((embeddings, full_embeddings), f)
    print(f"임베딩 생성 시간: {time.time() - start_time:.2f}초")

# DBSCAN (최적화: 유사도 행렬 캐싱)
def dbscan(db, eps, minPts):
    labels = len(db) * [None]
    c = 0

    # 코사인 유사도 행렬 미리 계산
    db_vectors = np.array([item[1] for item in db])
    similarity_matrix = cosine_similarity(db_vectors)

    for p in tqdm(range(len(db)), desc="DBSCAN"):
        if labels[p] is not None:
            continue

        neighborsN = np.where(similarity_matrix[p] >= eps)[0].tolist()
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
            neighborsQ = np.where(similarity_matrix[q] >= eps)[0].tolist()
            if len(neighborsQ) < minPts:
                continue
            seedSet.update(set(neighborsQ))

    return labels

# DBSCAN 실행
start_time = time.time()
eps = 0.003
minPts = 2
labels = dbscan(embeddings, eps, minPts)
print(f"DBSCAN 시간: {time.time() - start_time:.2f}초")

# 클러스터별 유형 식별 (author_id 기준 제거)
def identify_cluster_types(texts, labels, embeddings):
    cluster_texts = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append(embeddings[i][0])

    cluster_types = {}
    for cid, texts in cluster_texts.items():
        text_combined = " ".join(texts).lower()
        if "flight" in text_combined or "seat" in text_combined or "confirmation" in text_combined or "delay" in text_combined:
            cluster_types[cid] = "Flight Inquiry"
        elif "internet" in text_combined or "outage" in text_combined:
            cluster_types[cid] = "Service Outage"
        elif "bill" in text_combined or "payment" in text_combined:
            cluster_types[cid] = "Payment Issue"
        elif "shuffle" in text_combined or "repeat" in text_combined or "app" in text_combined:
            cluster_types[cid] = "App Issue"
        elif "order" in text_combined or "delivery" in text_combined:
            cluster_types[cid] = "Delivery Inquiry"
        elif "booking" in text_combined or "ticket" in text_combined:
            cluster_types[cid] = "Booking Issue"
        elif "battery" in text_combined or "charge" in text_combined:
            cluster_types[cid] = "Device Issue"
        else:
            cluster_types[cid] = "Other"
    return cluster_types

cluster_types = identify_cluster_types(texts, labels, embeddings)

# Silhouette Score 계산 (Noise 제외)
start_time = time.time()
valid_indices = [i for i, label in enumerate(labels) if label != -1]
valid_embeddings = full_embeddings[valid_indices]
valid_labels = [labels[i] for i in valid_indices]

if len(set(valid_labels)) > 1:
    silhouette_avg = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
    print(f"Silhouette Score: {silhouette_avg:.3f}")
else:
    print("Silhouette Score 계산 불가: 클러스터가 1개 또는 Noise만 존재.")
print(f"Silhouette Score 계산 시간: {time.time() - start_time:.2f}초")

# 데이터프레임 생성
plot_data = pd.DataFrame({
    'text': texts,
    'cluster': [str(label) for label in labels],
    'type': [cluster_types.get(label, "Noise") if label != -1 else "Noise" for label in labels],
    'author': authors
})

# 시각화
# TSNE로 2차원 축소
start_time = time.time()
n_samples = len(full_embeddings)
if n_samples < 10:
    print(f"샘플 크기가 너무 작습니다: {n_samples}. TSNE 시각화를 건너뜁니다.")
else:
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=300)
    embeddings_2d = tsne.fit_transform(full_embeddings)
    plot_data['x'] = embeddings_2d[:, 0]
    plot_data['y'] = embeddings_2d[:, 1]

    # 색상 맵 정의
    color_discrete_map = {
        'Noise': 'gray',
        'Flight Inquiry': 'blue',
        'Service Outage': 'orange',
        'Payment Issue': 'red',
        'App Issue': 'green',
        'Delivery Inquiry': 'purple',
        'Booking Issue': 'cyan',
        'Device Issue': 'pink',
        'Other': 'gray'
    }

    # Plotly Express로 2D 산점도 생성
    fig_2d = px.scatter(
        plot_data,
        x='x',
        y='y',
        color='type',
        hover_data=['text', 'type', 'author'],
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
    fig_2d.write_html("clustering_result8_2d.html")

    # 3D 시각화
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity, max_iter=300)
    embeddings_3d = tsne_3d.fit_transform(full_embeddings)
    plot_data['z'] = embeddings_3d[:, 2]

    # Plotly Express로 3D 산점도 생성
    fig_3d = px.scatter_3d(
        plot_data,
        x='x',
        y='y',
        z='z',
        color='type',
        hover_data=['text', 'type', 'author'],
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
    fig_3d.write_html("clustering_result8_3d.html")

print(f"시각화 시간: {time.time() - start_time:.2f}초")