import pandas as pd
from rouge import Rouge

from transformers import AutoTokenizer
import torch

from ML.train import Train


class Evaluation:

    def __init__(self):
        # v = Train()
        model = Train.loadTheTraindedModel()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Move model to correct device
        model = model.to(device)

        # input_text = "The cat was found in the garden and brought inside. It drank some milk and fell asleep"
        input_text = "Client: I am afraid to go to work."

        reference_text = ("Daniel is known for being punctual, highly organized, and emotionally intelligent."
                          " He performs well under pressure and fosters a respectful environment. "
                          "Colleagues appreciate his ability to listen and lead productive meetings. "
                          "Some mention that he can be overly analytical at times, which may slow down decision-making. "
                          "He gives honest feedback, though it can sometimes feel a bit too blunt."
                          "In group settings, his communication style could benefit from being more relaxed")

        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=True, is_trainable=True)
        embedings = tokenizer(reference_text, return_tensors="pt").to(device)
        text = model.generate(**embedings)
        print(tokenizer.decode(text[0], skip_special_tokens=True))
        tok = tokenizer.decode(text[0])
        print("Generated:", tok)
        print("Reference:", reference_text)
        rouge = Rouge()
        # Compute ROUGE
        eval_1_rouge = rouge.get_scores(reference_text, tok)
        rouge_scores_out = []

        for metric in ["rouge-1", "rouge-2", "rouge-l"]:
            for label in ["F-Score"]:
                eval_1_score = eval_1_rouge[0][metric][label[0].lower()]

                row = {
                    "Metric": f"{metric} ({label})",
                    "Summary 1": eval_1_score,
                }
                rouge_scores_out.append(row)

        def highlight_max(s):
            is_max = s == s.max()
            return [
                "background-color: lightgreen" if v else "background-color: white"
                for v in is_max
            ]

        rouge_scores_out = (
            pd.DataFrame(rouge_scores_out)
            .set_index("Metric")
            .style.apply(highlight_max, axis=1)
        )
        print(rouge_scores_out.to_excel)


if __name__ == '__main__':
    ev = Evaluation()
