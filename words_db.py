import threading

import numpy as np
import pandas as pd


class WordsDB:
    EXTRA_COLUMNS = ['translation', 'transcription', 'example_sentence', 'example_translation', 'conjugations', 'audio_path']

    def __init__(self, db_path):
        self.db_path = db_path
        self.words_df = pd.read_csv(self.db_path)
        self.words_df['id'] = self.words_df['id'].astype(int)
        if not self.words_df['id'].is_unique:
            raise ValueError('"id" field in the words database is not unique.')
        for col in self.EXTRA_COLUMNS:
            if col not in self.words_df.columns:
                self.words_df[col] = np.nan
        self._lock = threading.Lock()

    def save_words_db(self):
        self._lock.acquire()
        self.words_df.to_csv(self.db_path, index=False)
        self._lock.release()

    def get_words_df(self):
        self._lock.acquire()
        wdf_cpy = self.words_df.copy()
        self._lock.release()
        return wdf_cpy

    def get_word_data(self, word, lang):
        self._lock.acquire()
        res = self.words_df.loc[(self.words_df['word'] == word) & (self.words_df['lang'] == lang)].to_dict()
        self._lock.release()
        return res

    def add_new_word(self, word, lang, translation=None, transcription=None,
                     example_sentence=None, example_translation=None,
                     conjugations=None, audio_path=None):
        self._lock.acquire()
        word_data = self.words_df.loc[(self.words_df['lang'] == lang) & (self.words_df['word'] == word)]
        if word_data.shape[0] == 0:
            if self.words_df.empty or pd.isna(self.words_df['id'].max()):
                word_id = 0
            else:
                word_id = int(self.words_df['id'].max() + 1)
            new_row = {
                'id': word_id,
                'word': word,
                'lang': lang,
                'tags': np.nan,
                'meaning': np.nan,
                'translation': translation or np.nan,
                'transcription': transcription or np.nan,
                'example_sentence': example_sentence or np.nan,
                'example_translation': example_translation or np.nan,
                'conjugations': conjugations or np.nan,
                'audio_path': audio_path or np.nan
            }
            self.words_df.loc[len(self.words_df)] = new_row
        elif word_data.shape[0] == 1:
            word_id = int(word_data['id'].iloc[0])
            idx = word_data.index[0]
            if translation and pd.isna(self.words_df.at[idx, 'translation']):
                self.words_df.at[idx, 'translation'] = translation
            if transcription and pd.isna(self.words_df.at[idx, 'transcription']):
                self.words_df.at[idx, 'transcription'] = transcription
            if example_sentence and pd.isna(self.words_df.at[idx, 'example_sentence']):
                self.words_df.at[idx, 'example_sentence'] = example_sentence
            if example_translation and pd.isna(self.words_df.at[idx, 'example_translation']):
                self.words_df.at[idx, 'example_translation'] = example_translation
            if conjugations and pd.isna(self.words_df.at[idx, 'conjugations']):
                self.words_df.at[idx, 'conjugations'] = conjugations
            if audio_path and pd.isna(self.words_df.at[idx, 'audio_path']):
                self.words_df.at[idx, 'audio_path'] = audio_path
        else:
            self._lock.release()
            raise ValueError(f'The same word "{word}" appears >1 time in the database: {word_data}')
        self._lock.release()
        return word_id

    def release_lock(self):
        if self._lock.locked():
            self._lock.release()
