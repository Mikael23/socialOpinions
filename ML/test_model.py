
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from datasets import Dataset
from keras import device
from keras.src.callbacks import Callback
from matplotlib import pyplot as plt
from sklearn.metrics._scorer import metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, \
    PrinterCallback, ProgressCallback
import torch
from sklearn.model_selection import train_test_split
from Utils.data_sets_loader import DataSetsLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import losswise

class TestModel:

    print({"model "}, torch.backends.mps.is_available())  # should print: True

    model_name = 'google/flan-t5-base'
    a = DataSetsLoader()
    ds = a.load
    ds = ds.rename(columns={"Context": "input", "Response": "target"})
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    ds = Dataset.from_pandas(ds)
    input_text = "Client: I am afraid to go to work."

    reference_text = ("Daniel is known for being punctual, highly organized, and emotionally intelligent."
                      " He performs well under pressure and fosters a respectful environment. "
                      "Colleagues appreciate his ability to listen and lead productive meetings. "
                      "Some mention that he can be overly analytical at times, which may slow down decision-making. "
                      "He gives honest feedback, though it can sometimes feel a bit too blunt."
                      "In group settings, his communication style could benefit from being more relaxed")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=True, is_trainable=True)
    embedings = tokenizer(reference_text, return_tensors="pt").to(device)
    lora_config = LoraConfig(
        r=16,  # RANK
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T%
    )
    peft_model = get_peft_model(original_model, lora_config)
    text = peft_model.generate(**embedings)
    print(tokenizer.decode(text[0], skip_special_tokens=True))
    tok = tokenizer.decode(text[0])
    print("Generated:", tok)
    print("Reference:", reference_text)




if __name__ == '__main__':
    v = TestModel()