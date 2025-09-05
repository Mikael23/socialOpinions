from BD.bd_utils import BD
from Utils.preprocessor import TextProcessing
from data.surveus_seed import surveys
from pipelines import runIngestion

if __name__ == '__main__':
    print('test')
    store = BD()
    proc = TextProcessing()
    runIngestion(surveys, proc, store)
