import hashlib
import sqlite3
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from ML.summarize_opinions import summarize_opinions
from project_requests import SurveyResponseIn


class BD:
    collection = None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    _client = chromadb.PersistentClient(
        path="/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine/BD/chroma_db",
        settings=Settings(allow_reset=True))
    _collection = _client.get_or_create_collection(name="opinions", metadata={"hnsw:space": "cosine"})

    def _stable_id(self, text: str, scope: str) -> str:
        h = hashlib.sha1((str(scope) + "||" + text[:2048]).encode("utf-8")).hexdigest()
        return f"{scope}_{h}"

    # the method embed_texts is responsible for turning each chunk of text into a numeric vector (embedding)
    # using your chosen embedding model (in your case it looks like self.EMBED_MODEL_NAME).
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def upsert_chunks(self, chunks: List[str], base_meta: dict):
        if not chunks:
            return
        embeddings = self.embed_texts(chunks)
        ids = [self._stable_id(ch, base_meta.get("id")) for ch in chunks]

        metadatas = [{**{k: v for k, v in base_meta.items() if k != "comment"},
                      "chunkIndex": i,
                      "len": len(ch),
                      "model": self.EMBED_MODEL_NAME}
                     for i, ch in enumerate(chunks)]

        self._collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    def searchBySubject(self, surveyId: int = None,  k: int = 20):
        where = {"surveyId": surveyId}

        res = self._collection.get(
            where=where,
            limit=k,
            include=["documents", "metadatas"]
        )
        comments = res["documents"]
        return summarize_opinions(comments)

    def searchAuthorAboutSubject(self, col, query: str, authorUserId: str, subjectUserId: str, k: int = 10):
        res = col.query(
            query_texts=[query],
            n_results=min(k * 4, 100),
            where={"authorUserId": authorUserId, "subjectUserId": subjectUserId},
            include=["documents", "metadatas", "ids", "distances"]
        )
        return summarize_opinions(h["text"] for h in self._flatten(res))

    def searchSurveyByAuthor(self, col, query: str, surveyId: str, authorUserId: str, k: int = 10):
        res = col.query(
            query_texts=[query],
            n_results=min(k * 4, 100),
            where={"surveyId": surveyId, "authorUserId": authorUserId},
            include=["documents", "metadatas", "ids", "distances"]
        )
        return self._flatten(res)[:k]

    def _flatten(self, res):
        hits = []
        if res.get("ids") and res["ids"]:
            ids = res["ids"][0];
            docs = res["documents"][0];
            metas = res["metadatas"][0];
            for i in range(len(ids)):
                hits.append({"id": ids[i], "text": docs[i], "meta": metas[i], "score": 1 - dists[i]})
        return hits

    def init_vector_bd(self, bd_path: str):
        client = chromadb.Client(Settings(
            persist_directory="/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine/BD/chroma_db",
            # Saves data to disk
            anonymized_telemetry=False
        ))

        BD.collection = client.get_or_create_collection(name="opinions")
        # con = sqlite3.connect(bd_path)
        # cursor = con.cursor()
        # cursor.execute('''CREATE TABLE IF NOT EXISTS vectors (
        #                     token TEXT PRIMARY KEY,
        #                     embedding BLOB
        #                   )''')
        # con.commit()
        # con.close()

    # long
    # id,
    # long
    # surveyId,
    # long
    # areaId,
    # long
    # userId,
    # String
    # message,
    # LocalDateTime
    # localDateTime

    def get_doc(self):
        return BD.collection.get(include=["documents", "metadatas", "embeddings"])

    def find_doc(self, tokens: List[str]):
        embeddings = self.model.encode(tokens).tolist()
        if isinstance(embeddings, list):
            pass  # Already a list
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

            # Ensure embeddings are 2D (list of lists)
        if not isinstance(embeddings[0], list):
            embeddings = [embeddings]
        res = BD.collection.query(query_embeddings=embeddings, n_results=10)
        for doc in res['documents'][0]:
            c = self.model.encode(doc)

            c = np.expand_dims(c, axis=0)
            sim = cosine_similarity(embeddings, c)
            print(sim, doc)


if __name__ == '__main__':
    print(BD.get_doc())
    print(BD.init_vector_bd('the YAN  123123 /׳ק/׳ק3424 23גשדגעכגעכג ,   ,   '))
