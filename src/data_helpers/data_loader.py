import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import toolz as tlz
from src.data_helpers.utils import normalize_data
from src.data_helpers.prepare_dialogues import dialogues4movie


class MoviesDialoguesDataset(Dataset):
    def __init__(self, cornell_corpus_path, movie_name):
        self.questions, self.answers = dialogues4movie(cornell_corpus_path, movie_name)
        self.questions = pd.DataFrame(normalize_data(self.questions.values))
        self.answers = pd.DataFrame(normalize_data(self.answers.values))

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        if type(idx) == slice:
            input_ = self.questions[idx].to_list()
            target_ = self.answers[idx].to_list()
        else:
            input_ = [self.questions[idx]]
            target_ = [self.answers[idx]]

        sample = {
            'input': input_,
            'target': target_
        }
        return sample

    def create_lexicon(self):

        lexicon = set()

        def spliter(x: str): return x.split()

        def creator(x): lexicon.update(x)

        sentences: np.ndarray = pd.concat([self.answers, self.questions]).values

        list(tlz.thread_last(sentences,
                             normalize_data,
                             (map, spliter),
                             (map, creator)))

        return list(lexicon)
