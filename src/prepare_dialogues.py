import os
import ast
import pandas as pd

SEP = ' +++$+++ '

__COLUMNS_DICT = {
    'movie_titles_metadata.txt': ['movieID', 'title', 'year', 'IMDB rating', 'no. IMDB votes', 'genres'],
    'movie_characters_metadata.txt': ['characterID', 'character', 'name', 'movieID', 'movie', 'title', 'gender', 'pos'],
    'movie_lines.txt': ['lineID', 'characterID', 'movieID', 'character name', 'text of the utterance'],
    'movie_conversations.txt': ['1st_characterID', '2nd_characterID', 'movieID', 'utterances']
}


def cornell2pd(path):
    columns = __COLUMNS_DICT[os.path.split(path)[-1]]
    cornell_file = open(path, 'r', encoding='iso-8859-1').readlines()
    cornell_lists = list(map(lambda x: x.strip().split(SEP), cornell_file))
    df = pd.DataFrame(cornell_lists, columns=columns)
    return df


def dialogues4movie(cornell_corpus_data_root: str, movie_title: str = 'all'):
    if movie_title == 'all':
        movie_title = ''

    movies_titles: pd.DataFrame = cornell2pd(os.path.join(cornell_corpus_data_root, 'movie_titles_metadata.txt'))
    moviesID: list = movies_titles['movieID'][movies_titles['title'].str.contains(movie_title)].to_list()
    movie_lines: pd.DataFrame = cornell2pd(os.path.join(cornell_corpus_data_root, 'movie_lines.txt'))
    movie_conversations: pd.DataFrame = cornell2pd(os.path.join(cornell_corpus_data_root, 'movie_conversations.txt'))
    movie_conversations = movie_conversations[movie_conversations['movieID'].isin(moviesID)]
    movie_conversations['utterances'] = movie_conversations['utterances'].apply(lambda x: ast.literal_eval(x))

    question = []
    answer = []

    for conversation in movie_conversations['utterances']:
        if not len(conversation) % 2 == 0:
            conversation = conversation[:-1]
        for cnt, lineID in enumerate(conversation):
            if cnt % 2 == 0:
                question.append(lineID)
            else:
                answer.append(lineID)

    question_df = movie_lines['text of the utterance'][movie_lines['lineID'].isin(question)].reset_index(drop=True)
    answer_df = movie_lines['text of the utterance'][movie_lines['lineID'].isin(answer)].reset_index(drop=True)

    return question_df, answer_df
