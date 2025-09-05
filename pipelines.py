
from typing import Dict, List

from BD.bd_utils import BD
from Utils.preprocessor import TextProcessing


def runIngestion(surveys: List[Dict], proc: TextProcessing, store: BD):
    for survey in surveys:
        preprocessed = proc.preprocess(survey["text"])
        chunks = proc.chunk_text(preprocessed)
        baseMeta = {k: v for k, v in survey.items() if k != "text"}
        store.upsert_chunks(chunks, baseMeta)