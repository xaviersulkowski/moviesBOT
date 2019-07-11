from torch.utils.data import Dataset

from src.data_helpers.prepare_dialogues import dialogues4movie


class MoviesDialoguesDataset(Dataset):
    def __init__(self, cornell_corpus_path, movie_name):
        self.questions, self.answers = dialogues4movie(cornell_corpus_path, movie_name)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        sample = {
            'input': self.questions[idx],
            'target': self.answers[idx]
        }
        return sample
