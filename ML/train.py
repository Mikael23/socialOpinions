#########################
# Load the model
###############################
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
from livelossplot import PlotLossesKeras


class EpochLossDisplay(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"x {state.epoch}: Loss = {state.epoch['loss']}")


class Train:

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', use_fast=True, is_trainable=True)

    def compute_metrics(self,eval_pred):
        predictions, labels = eval_predS
        predictions = predictions[:, 0]
        print('HEre we are!!!!!!')
        return metric.compute(predictions=predictions, references=labels)

    def tokenize_function(self,batch):
        model_inputs = self.tokenizer(
            batch["input"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # Tokenize targets (manually if not a seq2seq model with built-in support)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch["target"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __init__(self):
        print({"model "} , torch.backends.mps.is_available())  # should print: True

        model_name = 'google/flan-t5-base'
        a = DataSetsLoader()
        ds = a.load
        ds = ds.rename(columns={"Context": "input", "Response": "target"})
        original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        ds = Dataset.from_pandas(ds)

        tokenized_dataset = ds.map(self.tokenize_function, batched=True)
        train_test_split_param = tokenized_dataset.train_test_split(test_size=0.3)
        test_valid = train_test_split_param['test'].train_test_split(test_size=0.1)


        # split_dataset = tokenized_datasets.train_test_split(test_size=0.3)
        print((ds[0]))

        lora_config = LoraConfig(
            r=16,  # RANK
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T%
        )
        peft_model = get_peft_model(original_model, lora_config)
        print(self.print_number_of_trainable_model_parameters(original_model))

        peft_training_args = TrainingArguments(
            output_dir="/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine",
            per_device_train_batch_size=1,
            evaluation_strategy="epoch",
            eval_do_concat_batches=False,
            logging_strategy='epoch',
            learning_rate=1e-3,  # higher learning rate than full fine-tuning
            num_train_epochs=100,  # increase for more accuracy
            logging_steps=1,
            use_mps_device=True,
            report_to="none"  # else it will ask for https://wandb.ai/ api
        )
        peft_training_args.set_logging(strategy="epoch")

        peft_trainer = Trainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=train_test_split_param['train'],
            eval_dataset=test_valid['train'],
        )

        history = peft_trainer.train()
        print(pd.DataFrame(peft_trainer.state.log_history))

        print(f"here we start!!!!!!!:{peft_trainer.state.log_history}")
        pd.DataFrame(peft_trainer.state.log_history).to_csv("log_history.csv", index=False)
        self.create_plot(peft_trainer.state.log_history)
        peft_model_path = "./peft-dialogue-summary-checkpoint-local"
        peft_model.save_pretrained(peft_model_path)

        # peft_trainer.model.save_pretrained(peft_model_path)

    def create_plot(self,log_history):
        epoch_data = defaultdict(dict)

        for entry in log_history:
            epoch = int(entry['epoch'])
            epoch_data[epoch].update(entry)

        # Sort by epoch
        epochs = sorted(epoch_data.keys())

        # Extract lists for plotting
        loss = [epoch_data[e].get('loss') for e in epochs]
        eval_loss = [epoch_data[e].get('eval_loss') for e in epochs]
        train_loss = [epoch_data[e].get('train_loss') for e in epochs]
        total_flos = [epoch_data[e].get('total_flos') for e in epochs]

        # Plotting
        plt.figure(figsize=(12, 8))

        plt.plot(epochs, loss, marker='o', label='loss')
        plt.plot(epochs, eval_loss, marker='o', label='eval_loss')
        plt.plot(epochs, train_loss, marker='o', label='train_loss')
        plt.plot(epochs, total_flos, marker='o', label='total_flos')

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics by Epoch')
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()

        return f"trainable model\n parameters:{trainable_model_params}\n all model parameters {all_model_params}\n percentrage of trainable model: {trainable_model_params / all_model_params * 100}"

    @staticmethod
    def loadTheTraindedModel():
        device1 = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",
                                                                torch_dtype=torch.float32).to(device1)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # original_model & instruct_model
        config = PeftConfig.from_pretrained('/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine/ML/peft-dialogue-summary-checkpoint-local')
        print("PEFT Adapter Config:", config)
        peft_model = PeftModel.from_pretrained(peft_model_base,
                                               '/Users/I552581/Library/CloudStorage/OneDrive-SAPSE/Documents/MachineLearning/BaseLine/ML/peft-dialogue-summary-checkpoint-local',
                                               torch_dtype=torch.float32,
                                               is_trainable=False)

        # the number of trainable parameters will be 0 due to is_trainable=False
        peft_model.eval()
        print("Loaded model class:", peft_model.__class__)

        peft_model.print_trainable_parameters()
        # print(print_number_of_trainable_model_parameters(peft_model))
        return peft_model

if __name__ == '__main__':
    v = Train()
