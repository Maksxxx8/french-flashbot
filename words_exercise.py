import math
import os
import random
import re
from typing import Any, List, Optional
import jinja2
import pandas as pd
from utils import get_assistant_response

from pydantic import BaseModel, Field, ValidationError, ConfigDict, model_validator

from exercise import Exercise


def normalize_french_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"['’`]", "'", s)
    s = re.sub(r'[^\w\s\']', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def sanitize_word_translation(translation: str, target_word: str) -> str:
    if not translation:
        return ""
    text = str(translation).strip()
    words_to_remove = [target_word.strip()]
    bare = re.sub(r"^(l'|le\s+|la\s+|un\s+|une\s+|les\s+|des\s+|d')", "", target_word, flags=re.IGNORECASE).strip()
    if bare and bare.lower() != target_word.lower():
        words_to_remove.append(bare)
    for w in words_to_remove:
        text = re.sub(r'[\(\[\{]\s*' + re.escape(w) + r'\s*[\)\]\}]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?i)\b' + re.escape(w) + r'\b', '', text)
    text = re.sub(r'[\(\[\{]\s*[\)\]\}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text or translation


class ExampleTestSentenceSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example_sentence: str
    sentence_translation: str
    difficulty: int = 1


class ExampleSentenceSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example_sentence: str
    sentence_translation: str
    pronunciation: Optional[str] = None


class WordTestSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example_list: list[ExampleTestSentenceSchema] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def wrap_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {'example_list': data}
        if isinstance(data, dict) and 'example_list' not in data:
            for k in ('examples', 'sentences', 'items', 'list'):
                if k in data and isinstance(data[k], list):
                    return {'example_list': data[k]}
        return data


class WordExamplesSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example_list: list[ExampleSentenceSchema] = Field(default_factory=list)
    pronunciation: Optional[str] = None   # IPA-транскрипция слова
    translation: Optional[str] = None     # Перевод слова на русский (для французского)
    conjugations: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def wrap_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {'example_list': data}
        if isinstance(data, dict) and 'example_list' not in data:
            for k in ('examples', 'sentences', 'items', 'list'):
                if k in data and isinstance(data[k], list):
                    data_cpy = dict(data)
                    data_cpy['example_list'] = data_cpy[k]
                    return data_cpy
        return data


class ResponseCorrectionSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    translation_score: int = 3
    score_justification: str = ""
    mistakes_explanation: Optional[str] = None
    corrected_translation: str = ""

    @model_validator(mode='before')
    @classmethod
    def extract_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            score = data.get('translation_score')
            if score is None:
                score = data.get('score', 3)
            justification = (data.get('score_justification') or data.get('justification') 
                             or data.get('feedback') or "")
            explanation = (data.get('mistakes_explanation') or data.get('explanation') 
                           or data.get('mistake_explanation') or "")
            corrected = (data.get('corrected_translation') or data.get('correct_translation') 
                         or data.get('corrected') or "")
            try:
                score_int = int(score)
            except Exception:
                score_int = 3
            return {
                'translation_score': score_int,
                'score_justification': str(justification),
                'mistakes_explanation': str(explanation) if explanation else None,
                'corrected_translation': str(corrected),
            }
        return data


class FlashCardExampleSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example: str = Field("", description="Example sentence")
    translation_of_example: str = Field("", description="Translate the example sentence")
    translation_of_word: str = Field("", description="Translate the word itself")


class FlashcardCorrectionSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    translation_score: int = 3
    score_justification: str = ""

    @model_validator(mode='before')
    @classmethod
    def extract_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            score = data.get('translation_score')
            if score is None:
                score = data.get('score', 3)
            justification = (data.get('score_justification') or data.get('justification') 
                             or data.get('feedback') or data.get('explanation') or "")
            try:
                score_int = int(score)
            except Exception:
                score_int = 3
            return {
                'translation_score': score_int,
                'score_justification': str(justification),
            }
        return data


class WordsExerciseLearn(Exercise):
    def __init__(self, word, word_id, lang, uilang, interface, templates,
                 meaning=None, translation=None, transcription=None,
                 example_sentence=None, example_translation=None, conjugations=None,
                 audio_path=None, num_reps=0, level='A1'):
        super().__init__()
        self.word = word
        self.meaning = meaning
        self.word_id = word_id
        self.lang = lang
        self.uilang = uilang
        self.level = level
        self.interface = interface
        self.templates = templates
        self.translation = translation
        self.transcription = transcription
        self.example_sentence = example_sentence
        self.example_translation = example_translation
        self.conjugations = conjugations
        self.audio_path = audio_path
        self.num_reps = num_reps + 1 if not math.isnan(num_reps) else 1
        self.model_base = os.getenv('MODEL_BASE', 'gemini-3.5-flash-lite')
        self.model_substitute = os.getenv('MODEL_SUBSTITUTE', 'gemini-3.5-flash-lite')

        self.is_responded = True

    async def get_next_user_message(self, user_response: Optional[str]) -> tuple[str, int]:
        message_template = self.templates.get_template(self.uilang, self.lang, 'learn_word_user_message')
        template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)

        # Если данные карточки уже предгенерированы — отдаём моментально!
        has_trans = self.translation and not pd.isna(self.translation)
        has_ex = self.example_sentence and not pd.isna(self.example_sentence)
        if has_trans and has_ex:
            ex_trans = str(self.example_translation) if self.example_translation and not pd.isna(self.example_translation) else ''
            example_list = [ExampleSentenceSchema(
                example_sentence=str(self.example_sentence),
                sentence_translation=ex_trans,
                pronunciation=None
            )]
            pron = str(self.transcription).strip().strip('[]') if self.transcription and not pd.isna(self.transcription) else None
            conj = str(self.conjugations) if self.conjugations and not pd.isna(self.conjugations) else None
            message = template.render(
                word=self.word,
                examples=example_list,
                conjugations=conj,
                pronunciation=pron,
                translation=str(self.translation)
            )
            return message, None

        # Fallback на генерацию через Gemini, если карточка не была предгенерирована:
        query_template = self.templates.get_template(self.uilang, self.lang, 'learn_word_query')
        q_temp = jinja2.Template(query_template, undefined=jinja2.StrictUndefined)
        word_phrase = "word" if len(self.word.split()) == 1 else "phrase"
        lang_tr = self.interface[self.lang][self.uilang]
        meaning = self.meaning if self.meaning and not (isinstance(self.meaning, float) and math.isnan(self.meaning)) else None
        query = q_temp.render(word_phrase=word_phrase, word=self.word, meaning=meaning, lang=lang_tr, level=self.level)

        schema = WordExamplesSchema.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"strict": True,
                            "name": "word_example",
                            "schema": schema
                            }
        }

        assistant_response = await get_assistant_response(self.interface, query, uilang=self.uilang, model_base=self.model_base,
                                                          model_substitute=self.model_substitute, response_format=response_format, validation_cls=WordExamplesSchema)
        
        message = template.render(word=self.word, examples=assistant_response.example_list,
                                  conjugations=assistant_response.conjugations,
                                  pronunciation=assistant_response.pronunciation,
                                  translation=assistant_response.translation)
        return message, None



class WordsExerciseTest(Exercise):
    def __init__(self, word, word_id, lang, uilang, level, interface, templates, known_words=None):
        super().__init__()
        self.word = word
        self.word_id = word_id
        self.lang = lang
        self.uilang = uilang
        self.level = level
        self.interface = interface
        self.templates = templates
        self.known_words = known_words or []
        self.n_examples = 1
        self.hint_clicked = False
        self.correct_answer_clicked = False
        self.is_responded = False
        self.difficulty = 2

        self.assistant_responses = []
        self.user_messages = []
        self.next_query_idx = 0
        self.model_base = os.getenv('MODEL_BASE', 'gemini-3.5-flash-lite')
        self.model_substitute = os.getenv('MODEL_SUBSTITUTE', 'gemini-3.5-flash-lite')

    def correct_answer(self):
        if not self.assistant_responses or not self.assistant_responses[0]:
            return self.word
        idx = max(0, min(len(self.assistant_responses[0]) - 1, self.difficulty - 1))
        return self.assistant_responses[0][idx]['answer']
    
    def test_sentence(self):
        if not self.assistant_responses or not self.assistant_responses[0]:
            return f"Переведите слово: {self.word}"
        idx = max(0, min(len(self.assistant_responses[0]) - 1, self.difficulty - 1))
        return self.assistant_responses[0][idx]['test']

    async def get_next_user_message(self, user_response: Optional[str]):
        lang_tr = self.interface[self.lang][self.uilang]
        if user_response is None:
            # first message to the user

            message_template = self.templates.get_template(self.uilang, self.lang, 'test_word_query_1')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            known_list = [w for w in (self.known_words or []) if w != self.word]
            known_words_str = ", ".join(known_list[:25]) if known_list else ""
            query = template.render(word=self.word, lang=lang_tr, level=self.level, known_words_str=known_words_str)

            validation_cls = WordTestSchema
            schema = validation_cls.model_json_schema()

            response_format = {
                "type": "json_schema",
                "json_schema": {"strict": True,
                                "name": "word_example",
                                "schema": schema
                                }
            }

            assistant_response = await get_assistant_response(self.interface, query, model_base=self.model_base,
                                                        model_substitute=self.model_substitute, uilang=self.uilang,
                                                        response_format=response_format, validation_cls=validation_cls)
            
            examples = sorted(assistant_response.example_list, key=lambda x: x.difficulty)
            examples = [dict(test=item.sentence_translation, answer=item.example_sentence) for item in examples]

            if not examples:
                examples = [dict(test=f"Переведите: {self.word}", answer=self.word)]

            self.difficulty = min(2, len(examples))

            self.assistant_responses.append(examples)

            message_template = self.templates.get_template(self.uilang, self.lang, 'test_word_user_message_1')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            message = template.render(lang=lang_tr, test_sentence=self.test_sentence())
            quality = None
            
        else:
            self.user_messages.append(user_response)
            clean_user = normalize_french_text(user_response or '')
            clean_target = normalize_french_text(self.correct_answer())

            if clean_user and clean_user == clean_target:
                assistant_response = ResponseCorrectionSchema(
                    translation_score=3,
                    score_justification="Отлично! Предложение переведено абсолютно точно.",
                    mistakes_explanation=None,
                    corrected_translation=self.correct_answer()
                )
            else:
                message_template = self.templates.get_template(self.uilang, self.lang, 'test_word_query_2')

                template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
                query = template.render(lang=self.interface[self.lang][self.uilang],
                                        user_response=user_response, sentence=self.test_sentence(),
                                        word=self.word)

                validation_cls = ResponseCorrectionSchema
                schema = validation_cls.model_json_schema()

                response_format = {
                    "type": "json_schema",
                    "json_schema": {"strict": True,
                                    "name": "word_example",
                                    "schema": schema
                                    }
                }

                assistant_response = await get_assistant_response(self.interface, query, model_base=self.model_base,
                                                            model_substitute=self.model_substitute, uilang=self.uilang,
                                                            response_format=response_format, validation_cls=validation_cls)
            message_template = self.templates.get_template(self.uilang, self.lang, 'test_word_user_message_2')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            message = template.render(score=assistant_response.translation_score,
                                  justification=assistant_response.score_justification,
                                  explanation=assistant_response.mistakes_explanation,
                                  corrected_translation=assistant_response.corrected_translation,
                                  original_translation=self.correct_answer())
            quality = assistant_response.translation_score

        return message, quality
    

    def change_difficulty(self, easier: bool) -> None:
        lang_tr = self.interface[self.lang][self.uilang]

        if easier:
            self.difficulty = max(1, self.difficulty - 1)
        else:
            self.difficulty = min(5, self.difficulty + 1)

        message_template = self.templates.get_template(self.uilang, self.lang, 'test_word_user_message_1')
        template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
        message = template.render(lang=lang_tr, test_sentence=self.test_sentence())
        return message


class FlashcardExercise(Exercise):
    def __init__(self, word, word_id, lang, uilang, level, interface, templates,
                 translation=None, example_sentence=None, example_translation=None, known_words=None):
        super().__init__()
        self.word = word
        self.word_id = word_id
        self.lang = lang
        self.uilang = uilang
        self.level = level
        self.interface = interface
        self.templates = templates
        self.translation = translation
        self.example_sentence = example_sentence
        self.example_translation = example_translation
        self.known_words = known_words or []
        self.n_examples = 1
        self.hint_clicked = False
        self.correct_answer_clicked = False
        self.is_responded = False

        self.assistant_responses = []
        self.user_messages = []
        self.next_query_idx = 0
        self.model_base = os.getenv('MODEL_BASE', 'gemini-3.5-flash-lite')
        self.model_substitute = os.getenv('MODEL_SUBSTITUTE', 'gemini-3.5-flash-lite')

    def correct_answer(self):
        ex = self.assistant_responses[0].get("example", "") if self.assistant_responses and self.assistant_responses[0] else ""
        return f'{self.word}\n\n{self.interface["Example"][self.uilang]}: {ex}'

    async def get_next_user_message(self, user_response: Optional[str]):
        lang_tr = self.interface[self.lang][self.uilang]
        if user_response is None:
            # Если перевод и пример уже есть в базе данных — используем их напрямую без LLM!
            has_data = (self.translation and not pd.isna(self.translation) and
                        self.example_sentence and not pd.isna(self.example_sentence) and
                        self.example_translation and not pd.isna(self.example_translation))
            if has_data:
                clean_trans = sanitize_word_translation(str(self.translation), self.word)
                self.assistant_responses.append(dict(
                    example=str(self.example_sentence),
                    translation_example=str(self.example_translation),
                    translation_word=clean_trans
                ))
            else:
                # first message to the user
                message_template = self.templates.get_template(self.uilang, self.lang, 'flashcard_query_1')
                template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
                query = template.render(word=self.word, level=self.level, lang=lang_tr, lang_ui=self.uilang)

                validation_cls = FlashCardExampleSchema
                schema = validation_cls.model_json_schema()

                response_format = {
                    "type": "json_schema",
                    "json_schema": {"strict": True,
                                    "name": "word_example",
                                    "schema": schema
                                    }
                }

                assistant_response = await get_assistant_response(self.interface, query, model_base=self.model_base,
                                                            model_substitute=self.model_substitute, uilang=self.uilang,
                                                            response_format=response_format, validation_cls=validation_cls)
                
                clean_trans = sanitize_word_translation(assistant_response.translation_of_word, self.word)
                self.assistant_responses.append(dict(example=assistant_response.example, translation_example=assistant_response.translation_of_example,
                                                     translation_word=clean_trans))

            word_tr = self.assistant_responses[0]['translation_word'] if self.assistant_responses and self.assistant_responses[0] else self.word
            ex_tr = self.assistant_responses[0]['translation_example'] if self.assistant_responses and self.assistant_responses[0] else ""
            message_template = self.templates.get_template(self.uilang, self.lang, 'flashcard_user_message_1')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            message = template.render(lang=lang_tr, lang_ui=self.uilang, word=word_tr, example=ex_tr)
            quality = None
            
        else:
            self.user_messages.append(user_response)
            clean_user = normalize_french_text(user_response or '')
            clean_target = normalize_french_text(self.word)
            bare_target = normalize_french_text(re.sub(r"^(l'|le\s+|la\s+|un\s+|une\s+|les\s+|des\s+|d')", "", self.word, flags=re.IGNORECASE))

            if clean_user and clean_user == clean_target:
                assistant_response = FlashcardCorrectionSchema(
                    translation_score=3,
                    score_justification="Отлично! Абсолютно верный перевод."
                )
            elif clean_user and bare_target and clean_user == bare_target:
                assistant_response = FlashcardCorrectionSchema(
                    translation_score=3,
                    score_justification=f"Правильно! Слово верно переведено. Не забывай артикль: {self.word}."
                )
            else:
                message_template = self.templates.get_template(self.uilang, self.lang, 'flashcard_query_2')

                template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
                word_tr = self.assistant_responses[-1]['translation_word'] if self.assistant_responses and self.assistant_responses[-1] else self.word
                query = template.render(lang=self.interface[self.lang][self.uilang], user_response=user_response,
                                        word_translation=word_tr, correct_answer=self.word)

                validation_cls = FlashcardCorrectionSchema
                schema = validation_cls.model_json_schema()

                response_format = {
                    "type": "json_schema",
                    "json_schema": {"strict": True,
                                    "name": "word_example",
                                    "schema": schema
                                    }
                }

                assistant_response = await get_assistant_response(self.interface, query, model_base=self.model_base,
                                                            model_substitute=self.model_substitute, uilang=self.uilang,
                                                            response_format=response_format, validation_cls=validation_cls)

            message_template = self.templates.get_template(self.uilang, self.lang, 'flashcard_user_message_2')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            correct_answer = self.word if assistant_response.translation_score < 5 else None
            context_translation = self.assistant_responses[-1]['example'] if self.assistant_responses and self.assistant_responses[-1] else ""
            message = template.render(score=assistant_response.translation_score,
                                justification=assistant_response.score_justification,
                                correct_answer=correct_answer,
                                context_translation=context_translation)
            quality = assistant_response.translation_score

        return message, quality
