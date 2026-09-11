import asyncio
import math
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional
import jinja2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from exercise import Exercise
from item import Item
from utils import get_assistant_response, get_audio
from words_exercise import FlashcardExercise, WordsExerciseLearn, WordsExerciseTest


class GeneratedWordItem(BaseModel):
    class Config:
        extra = 'forbid'

    word: str
    translation: str
    transcription: str
    example_sentence: str
    example_translation: str
    conjugations: Optional[str]


class NewWordsBatchSchema(BaseModel):
    class Config:
        extra = 'forbid'

    deck_theme: str
    words: list[GeneratedWordItem]


class LearningPlan:
    def __init__(self, interface, templates, words_progress_db=None, words_db=None, decks_db=None, user_config=None):
        self.progress_db = words_progress_db
        self.words_db = words_db
        self.decks_db = decks_db
        self.user_config = user_config
        self.interface = interface
        self.templates = templates
        self.max_n_reps = 10  # a word will not be tested more than this many times
        self._replenishing = set()
    
    def calculate_interval(self, item: Item) -> int:
        """Returns number of days until the next review."""
        if item.last_interval in [0, 1]:
            return 1
        elif item.last_interval == 2:
            return 6
        else:
            return math.ceil(item.last_interval * item.e_factor)
    
    def calculate_e_factor(self, item: Item, quality: int) -> float:
        new_ef = item.e_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        return max(1.3, new_ef)
    
    async def get_next_words_exercise(self, chat_id: str, lang: str, mode: Optional[str]=None) -> Optional[Exercise]:
        
        if mode == 'learn' and not self.has_enough_words(chat_id, lang):
            await self.add_words(chat_id, lang)

        now = datetime.now().date()

        words_df = self.words_db.get_words_df()
        deck_words_df = self.decks_db.get_deck_word_df()

        deck_words_df = pd.merge(words_df, deck_words_df, how='inner', left_on='id', right_on='word_id', sort=False)
        user_decks = self.decks_db.get_user_decks(chat_id, lang)
        deck_words_df = deck_words_df[deck_words_df['deck_id'].isin(user_decks)]

        if deck_words_df.shape[0] == 0:
            return None

        progress_df = self.progress_db.get_progress_df()
        last_review_date = pd.to_datetime(progress_df['last_review_date']).dt.date
        n_done_today = progress_df.loc[(progress_df['chat_id'] == chat_id) & (last_review_date == now)].shape[0]
        n_tests_done_today = progress_df.loc[(progress_df['chat_id'] == chat_id) & (last_review_date == now) & (progress_df['num_reps'] > 0)].shape[0]
        user_data = self.user_config.get_user_data(chat_id)
        n_flashcards = user_data.get('n_flashcards', 5)

        # reset progress on words that are long due (only for words that were already studied at least once)
        words_progress = pd.merge(progress_df, deck_words_df, how='left', left_on='word_id', right_on='word_id', sort=False)
        user_not_ignored_mask = ((words_progress['lang'] == lang.lower()) & (words_progress['chat_id'] == chat_id) & words_progress['to_ignore'].isin([False, np.nan]))
        long_due_today_mask = user_not_ignored_mask & (words_progress['num_reps'] > 0) & (words_progress['next_review_date'] <= now - timedelta(days=5)) & (words_progress['num_reps'] < self.max_n_reps)
        long_due_words = words_progress[long_due_today_mask]['word_id'].to_list()
        if len(long_due_words) > 0:
            self.progress_db.remove_progress(chat_id, long_due_words)
        self.progress_db.save_progress()
        progress_df = self.progress_db.get_progress_df()

        if mode is None:
            if len(long_due_words) > 0:
                mode = 'test_flashcard'
            elif n_done_today == 0:
                mode = 'learn'
            elif n_tests_done_today % n_flashcards == 0:
                mode = 'test_translation'
            else:
                mode = 'test_flashcard'

        if mode == 'test':
            mode = 'test_translation' if (n_tests_done_today > 0) and (n_tests_done_today % n_flashcards == 0) else 'test_flashcard'

        if mode in ['test_flashcard', 'test_translation']:
            progress_df = progress_df[progress_df['to_ignore'].isin([False, np.nan])]
            words_progress = pd.merge(progress_df, deck_words_df, how='left', left_on='word_id', right_on='word_id', sort=False)
            user_words_progress = words_progress.loc[(words_progress['lang'] == lang.lower()) & (words_progress['chat_id'] == chat_id)]

            if user_words_progress.shape[0] == 0:
                return None
            
            user_words_progress = user_words_progress.sort_values(by='next_review_date')

            # Исключаем слова, которые УЖЕ повторялись/тестировались сегодня
            last_review_dates = pd.to_datetime(user_words_progress['last_review_date']).dt.date
            reviewed_today_mask = (last_review_dates == now)

            # Слова для повторения: не повторялись сегодня, подошел срок (next_review_date <= now), и уже изучались (num_reps > 0)
            to_review_mask = (
                (~reviewed_today_mask) & 
                (user_words_progress['next_review_date'] <= now) & 
                (user_words_progress['num_reps'] < self.max_n_reps) &
                (user_words_progress['num_reps'] > 0)
            )

            if to_review_mask.sum() > 0:
                to_review_words = user_words_progress[to_review_mask]
                row_item = to_review_words.iloc[0]
            else:
                # Все запланированные на сегодня слова уже повторены!
                # Ни в коем случае не повторяем одно и то же слово дважды за день.
                return None
        else:
            user_progress = progress_df[progress_df['chat_id'] == chat_id] if progress_df.shape[0] > 0 else progress_df
            user_words_progress = pd.merge(user_progress, deck_words_df, how='right', left_on='word_id', right_on='id', sort=False)
            user_words_progress = user_words_progress[user_words_progress['to_ignore'].isin([False, np.nan])]

            unseen_words = user_words_progress[user_words_progress['last_review_date'].isna()]

            if unseen_words.shape[0] == 0:
                # Все слова в буфере уже изучены! Генерируем новую порцию из 10 слов
                await self.add_words(chat_id, lang)
                progress_df = self.progress_db.get_progress_df()
                words_df = self.words_db.get_words_df()
                deck_words_df = self.decks_db.get_deck_word_df()
                deck_words_df = pd.merge(words_df, deck_words_df, how='inner', left_on='id', right_on='word_id', sort=False)
                deck_words_df = deck_words_df[deck_words_df['deck_id'].isin(user_decks)]
                user_progress = progress_df[progress_df['chat_id'] == chat_id] if progress_df.shape[0] > 0 else progress_df
                user_words_progress = pd.merge(user_progress, deck_words_df, how='right', left_on='word_id', right_on='id', sort=False)
                user_words_progress = user_words_progress[user_words_progress['to_ignore'].isin([False, np.nan])]
                unseen_words = user_words_progress[user_words_progress['last_review_date'].isna()]

            if unseen_words.shape[0] > 0:
                row_item = unseen_words.iloc[0]
            else:
                return None

        user_level = self.user_config.get_user_data(chat_id)['level']
        uilang = self.user_config.get_user_ui_lang(chat_id)
                
        def _to_scalar(val, default=0):
            if val is None or pd.isna(val):
                return default
            if hasattr(val, 'item'):
                try:
                    return val.item()
                except Exception:
                    pass
            return val

        def _clean_str(val):
            if val is None or pd.isna(val):
                return None
            s = str(val).strip()
            if s.lower() == 'nan' or not s:
                return None
            return s

        raw_id = row_item['id'] if 'id' in row_item else row_item['word_id']
        word_id = int(_to_scalar(raw_id, 0))
        num_reps = float(_to_scalar(row_item['num_reps'] if 'num_reps' in row_item else 0.0, 0.0))
        meaning = _clean_str(row_item.get('meaning'))

        translation = _clean_str(row_item.get('translation'))
        transcription = _clean_str(row_item.get('transcription'))
        example_sentence = _clean_str(row_item.get('example_sentence'))
        example_translation = _clean_str(row_item.get('example_translation'))
        conjugations = _clean_str(row_item.get('conjugations'))
        audio_path = _clean_str(row_item.get('audio_path'))

        if 'test_flashcard' == mode:
            exercise = FlashcardExercise(word=row_item['word'], word_id=word_id, lang=lang, uilang=uilang, level=user_level,
                                         interface=self.interface, templates=self.templates)
        elif 'test_translation' == mode:
            exercise = WordsExerciseTest(word=row_item['word'], word_id=word_id, lang=lang, uilang=uilang, level=user_level,
                                         interface=self.interface, templates=self.templates)
        else:
            exercise = WordsExerciseLearn(word=row_item['word'], meaning=meaning,
                                          translation=translation, transcription=transcription,
                                          example_sentence=example_sentence, example_translation=example_translation,
                                          conjugations=conjugations, audio_path=audio_path,
                                          word_id=word_id, lang=lang, uilang=uilang,
                                          num_reps=num_reps, interface=self.interface,
                                          templates=self.templates, level=user_level)
            # Фоновая проверка и пополнение буфера (если в запасе осталось <= 5 слов, добавит +10 в фоне)
            asyncio.create_task(self.ensure_word_buffer(chat_id, lang))

        return exercise

    def get_due_today(self, chat_id: str, lang: str) -> List[str]:

        now = datetime.now().date()

        words_df = self.words_db.get_words_df()
        deck_words_df = self.decks_db.get_deck_word_df()

        deck_words_df = pd.merge(words_df, deck_words_df, how='inner', left_on='id', right_on='word_id', sort=False)
        user_decks = self.decks_db.get_user_decks(chat_id, lang)
        deck_words_df = deck_words_df[deck_words_df['deck_id'].isin(user_decks)]

        if deck_words_df.shape[0] == 0:
            return None

        progress_df = self.progress_db.get_progress_df()

        progress_df = progress_df[progress_df['to_ignore'].isin([False, np.nan])]
        words_progress = pd.merge(progress_df, deck_words_df, how='left', left_on='word_id', right_on='word_id', sort=False)
        user_words_progress = words_progress.loc[(words_progress['lang'] == lang.lower()) & (words_progress['chat_id'] == chat_id)]

        if user_words_progress.shape[0] == 0:
            return None
    
        user_words_progress = user_words_progress.sort_values(by='next_review_date')
        last_reviewed_dates = pd.to_datetime(user_words_progress['last_review_date']).dt.date
        reviewed_today_mask = (last_reviewed_dates == now)
        to_review_mask = (
            (~reviewed_today_mask) & 
            ((user_words_progress['next_review_date'] <= now) | user_words_progress['next_review_date'].isna()) & 
            (user_words_progress['num_reps'] < self.max_n_reps)
        )
        if to_review_mask.sum() > 0:
            to_review_words = user_words_progress[to_review_mask]['word'].to_list()
        else:
            to_review_words = []

        return to_review_words


    def process_hint(self, chat_id: int, exercise: Exercise) -> None:
        item = self.progress_db.get_word_progress(chat_id, exercise.word_id)
        item.num_reps = 1
        item.last_interval = max(math.floor(item.last_interval / 2), 0)
        item.last_review_date = datetime.now()
        item.next_review_date = (datetime.now() + timedelta(days=1)).date()
        self.progress_db.set_word_progress(chat_id, exercise.word_id, item)

    def process_correct_answer(self, chat_id: int, exercise: Exercise) -> None:
        item = self.progress_db.get_word_progress(chat_id, exercise.word_id)
        item.num_reps = 1
        item.last_interval = 0
        item.last_review_date = datetime.now()
        item.next_review_date = (datetime.now() + timedelta(days=1)).date()
        self.progress_db.set_word_progress(chat_id, exercise.word_id, item)

    def process_response(self, chat_id: int, exercise: Exercise, quality: Optional[int]) -> None:

        item = self.progress_db.get_word_progress(chat_id, exercise.word_id)

        if quality is None:
            # a word was learned
            if item is None:
                self.progress_db.add_word_to_progress(chat_id, exercise.word_id)
            item = self.progress_db.get_word_progress(chat_id, exercise.word_id)
            now = datetime.now()
            item.num_reps = 1
            item.last_interval = 0
            item.last_review_date = now
            item.next_review_date = (now + timedelta(days=1)).date()
        elif quality is not None:
            # a word got tested

            if exercise.correct_answer_clicked or exercise.hint_clicked:
                # last interval and next review date are already updated
                return

            if not 0 <= quality <= 5:
                raise ValueError("Quality must be between 0 and 5")

            item.e_factor = self.calculate_e_factor(item, quality)

            if quality < 3:
                item.num_reps = 1
                item.last_interval = 0
            else:
                item.num_reps += 1
            
            new_interval = self.calculate_interval(item)
            item.last_interval += new_interval
            now = datetime.now()
            item.last_review_date = now
            item.next_review_date = (now + timedelta(days=new_interval)).date()

        self.progress_db.set_word_progress(chat_id, exercise.word_id, item)

    def get_unseen_count(self, chat_id: str, lang: str) -> int:
        """Возвращает количество готовых неизученных слов в запасе."""
        progress_df = self.progress_db.get_progress_df()
        words_df = self.words_db.get_words_df()
        deck_words_df = self.decks_db.get_deck_word_df()
        user_decks = self.decks_db.get_user_decks(chat_id, lang)

        deck_words_df = pd.merge(words_df, deck_words_df, how='inner', left_on='id', right_on='word_id', sort=False)
        deck_words_df = deck_words_df[deck_words_df['deck_id'].isin(user_decks)]

        if deck_words_df.shape[0] == 0:
            return 0

        user_progress = progress_df[progress_df['chat_id'] == chat_id] if progress_df.shape[0] > 0 else progress_df
        user_words_progress = pd.merge(user_progress, deck_words_df, how='right', left_on='word_id', right_on='id', sort=False)
        user_words_progress = user_words_progress[user_words_progress['to_ignore'].isin([False, np.nan])]

        unseen = user_words_progress[user_words_progress['last_review_date'].isna()]
        return unseen.shape[0]

    async def ensure_word_buffer(self, chat_id: str, lang: str):
        """
        Гарантирует, что в запасе всегда от 5 до 15 слов.
        Если в запасе <= 5 слов, генерирует +10 новых слов в фоне.
        """
        key = (str(chat_id), lang)
        if key in self._replenishing:
            return
        self._replenishing.add(key)
        try:
            unseen_count = self.get_unseen_count(chat_id, lang)
            print(f'[Buffer] Проверка запаса слов: сейчас {unseen_count} неизученных слов.')
            if unseen_count <= 5:
                print(f'[Buffer] Запас неизученных слов ({unseen_count}) <= 5. Запускаю фоновую догенерацию +10 слов...')
                await self.add_words(chat_id, lang)
                new_count = self.get_unseen_count(chat_id, lang)
                print(f'[Buffer] Пополнение буфера завершено. Теперь в запасе: {new_count} слов.')
        except Exception as e:
            print(f'[Buffer] Ошибка при фоновом пополнении буфера: {e}')
        finally:
            self._replenishing.discard(key)

    def has_enough_words(self, chat_id, lang):
        return self.get_unseen_count(chat_id, lang) > 0

    async def add_words(self, chat_id, lang):
        user_data = self.user_config.get_user_data(chat_id)
        uilang = user_data['ui_language']

        words_df = self.words_db.get_words_df()
        progress_df = self.progress_db.get_progress_df()

        if words_df.shape[0] > 0 and progress_df.shape[0] > 0:
            progress_words_df = pd.merge(progress_df, words_df, how='outer', left_on='word_id',
                                         right_on='id', sort=False)
            not_ignored_words = progress_words_df.loc[(progress_words_df['lang'] == lang.lower()) & (progress_words_df['chat_id'] == chat_id) &
                                                      progress_words_df['to_ignore'].isin([False, np.nan])]
            if not_ignored_words.shape[0] > 0:
                user_words_str = ', '.join(not_ignored_words['word'].to_list())
            else:
                user_words_str = 'No words learned yet.'
        else:
            user_words_str = 'No words learned yet.'

        message_template = self.templates.get_template(uilang, lang, 'gen_words')
        template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
        query = template.render(lang=lang, user_words_str=user_words_str)

        model_base = os.getenv('MODEL_BASE', 'gemini-3.6-flash')
        model_substitute = os.getenv('MODEL_SUBSTITUTE', 'gemini-3.6-flash')

        validation_cls = NewWordsBatchSchema
        schema = validation_cls.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"strict": True,
                            "name": "new_words_batch",
                            "schema": schema
                            }
        }

        generated_items = []
        for i in range(3):
            assistant_response = await get_assistant_response(
                self.interface, query, model_base=model_base,
                model_substitute=model_substitute, uilang=user_data['ui_language'],
                response_format=response_format, validation_cls=validation_cls
            )
            words_list = assistant_response.words
            new_items = [item for item in words_list if words_df[(words_df['word'].str.lower() == item.word.lower()) & (words_df['lang'] == lang)].shape[0] == 0]
            if len(new_items) >= 5:
                generated_items = new_items
                break
            elif len(new_items) > len(generated_items):
                generated_items = new_items

        if not generated_items:
            print('[Buffer] Предупреждение: Не удалось сгенерировать новые уникальные слова.')
            return

        custom_deck_id = self.decks_db.get_custom_deck_id(str(chat_id), lang)
        if custom_deck_id is None:
            custom_deck_id = self.decks_db.add_custom_deck(str(chat_id), lang)

        audio_dir = Path('resources/audio')
        audio_dir.mkdir(parents=True, exist_ok=True)

        for item in generated_items:
            safe_name = re.sub(r'[^a-zA-Zа-яА-Я0-9_]', '_', item.word)
            audio_path = str(audio_dir / f"{safe_name}_{int(time.time()*1000)}.mp3")
            try:
                await get_audio(item.word, lang, audio_path)
            except Exception as e:
                print(f'[Buffer] Ошибка предгенерации аудио для "{item.word}": {e}')
                audio_path = None

            word_id = self.words_db.add_new_word(
                word=item.word,
                lang=lang,
                translation=item.translation,
                transcription=item.transcription,
                example_sentence=item.example_sentence,
                example_translation=item.example_translation,
                conjugations=item.conjugations,
                audio_path=audio_path
            )
            self.decks_db.add_new_word(custom_deck_id, word_id)

        self.words_db.save_words_db()
        self.decks_db.save_decks_db()
        print(f'[Buffer] Успешно добавлено {len(generated_items)} новых слов в буфер (колода {custom_deck_id}).')

