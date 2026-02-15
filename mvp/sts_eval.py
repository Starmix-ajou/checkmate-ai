import nltk
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

model = SentenceTransformer("all-MiniLM-L6-v2")

client = MongoClient("mongodb://localhost:27017/")
db = client["test_db"]
collection = db["vectors"]

async def store_sentences_as_vetors(json_obj, doc_id):
    text = await extract_text_from_json(json_obj)
    sentences = sent_tokenize(text)
    vectors = model.encode(sentences)
    collection.insert_one({"_id": doc_id, "text": sentences, "embedding": vectors.tolist()})

async def extract_text_from_json(json_obj):
    text_obj = json_obj["description"]
    return " ".join([str(v) for v in text_obj.values() if isinstance(v, str)])

async def find_similar_vector(vector):
    query = {
        "index": "default",
        "pipeline": [],
        "queryVector": vector.tolist(),
        "path": "embedding",
        "numCandidates": 50,
        "limit": 1
    }
    result = db.command({
        "aggregate": collection.name,
        "pipeline": [
            {
                "$vectorSearch": query
            }
        ],
        "cursor": {}
    })
    docs = list(result['cursor']['firstBatch'])
    if docs:
        return docs[0]["embedding"], docs[0]["text"]
    else:
        return None, None

async def compute_sts_between_json_objects(json_obj1, json_obj2):
    collection.delete_many({})
    await store_sentences_as_vetors(json_obj1, doc_id="json_obj2")
    
    text1 = await extract_text_from_json(json_obj1)
    sentence1 = sent_tokenize(text1)
    vector1 = model.encode(sentence1)
    
    scores = []
    for sent, vec in zip(sentence1, vector1):
        match_vec, match_text = await find_similar_vector(vec)
        if match_text:
            score = cosine_similarity([vec], [match_vec])[0][0]
            scores.append(score)
            
    if scores:
        return np.mean(scores)
    else:
        return 0.0

json_a = {
    "title": "AI 기반 회의 요약",
    "description": "이 시스템은 회의 내용을 분석하여 요약을 제공합니다."
}
json_b = {
    "title": "자동 요약 시스템",
    "description": "우리 모델은 회의록을 분석하고 핵심 내용을 정리합니다."
}

average_sts = compute_sts_between_json_objects(json_a, json_b)
print(f"Average STS Score: {average_sts:.4f}")