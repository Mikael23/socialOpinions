import hashlib
import re
import sqlite3
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class TextProcessing:

    def preprocess(self, input_text: str):
        text = input_text.replace("\r", "\n")
        text = re.compile(r"\s+").sub(" ", text).strip()
        return text

    def chunk_text(self, text: str, chunk_size: int = 160, overlap: int = 30) -> List[str]:
        words = text.split()
        out = []
        i = 0
        while i < len(words):
            out.append(" ".join(words[i:i + chunk_size]))
            if i + chunk_size >= len(words):
                break
            i += max(1, chunk_size - overlap)
        return out

    def tokenizer(self, text: str):
        return text.split()


if __name__ == '__main__':
    print(TextProcessing.preprocess('the YAN  123123 /׳ק/׳ק3424 23גשדגעכגעכג ,   ,   '))
    pre = TextProcessing.preprocess('the YAN  123123 /׳ק/׳ק3424 23גשדגעכגעכג ,   ,   ')



# Why do we chunk text before embedding!!!!!!!!!!!!!!!!?
#
# Imagine a long survey response or comment like:
#
# "I went to the gym yesterday, the equipment was broken, the trainer was helpful though,
# and the shower was dirty. In the café afterwards the food was decent but overpriced…"
#
# If you send this whole block (say 1,000+ tokens) to your embedding model:
#
# The vector will be a blurry average of all topics (gym, trainer, shower, café).
#
# If a user asks: “How is the shower?” → the similarity score might be diluted, and retrieval could fail.
#
# Instead we chunk into smaller, focused pieces:
#
# "I went to the gym yesterday, the equipment was broken."
#
# "The trainer was helpful though."
#
# "The shower was dirty."
#
# "In the café afterwards the food was decent but overpriced."
#
# Now each chunk has a clear semantic meaning, and embeddings capture that.
# So for “How is the shower?”, retrieval will directly pull chunk #3.



# _stable_id!!!!!!!!!!!! gives you a deterministic, reproducible, unique key per chunk,
# so your upsert operation in the vector DB always does the right thing: overwrite, insert,
# or delete consistently.