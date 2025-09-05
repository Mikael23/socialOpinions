from datasets import load_dataset
from tensorboard.notebook import display
import pandas as pd


class DataSetsLoader:



    @property
    def load(self):
        pd.set_option('display.max_colwidth', None)
        df = pd.read_json("https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/resolve/main/combined_dataset.json", lines=True)
        print(df.head())
        # print(df.columns)

        return df

if __name__ == '__main__':
   a= DataSetsLoader()
   a.load



