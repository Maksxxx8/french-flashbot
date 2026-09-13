import math
import os
import random
import re
from typing import List, Optional
import jinja2
import pandas as pd
from utils import get_assistant_response

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from exercise import Exercise


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


class WordExamplesSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example_list: list[ExampleSentenceSchema] = Field(default_factory=list)
    pronunciation: Optional[str] = None   # IPA-транскрипция слова
    translation: Optional[str] = None     # Перевод слова на русский (для французского)
    conjugations: Optional[str] = None


class ResponseCorrectionSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    translation_score: int = 3
    score_justification: str = ""
    mistakes_explanation: Optional[str] = None
    corrected_translation: str = ""


class FlashCardExampleSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    example: str = Field("", description="Example sentence")
    translation_of_example: str = Field("", description="Translate the example sentence")
    translation_of_word: str = Field("", description="Translate the word itself")


class FlashcardCorrectionSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')

    translation_score: int = 3
    score_justification: str = ""


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
        idx = max(0, min(len(self.assistant_responses[0]) - 1, self.difficulty - 1))
        return self.assistant_responses[0][idx]['answer']
    
    def test_sentence(self):
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

            self.difficulty = 2

            self.assistant_responses.append(examples)

            message_template = self.templates.get_template(self.uilang, self.lang, 'test_word_user_message_1')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            message = template.render(lang=lang_tr, test_sentence=self.test_sentence())
            quality = None
            
        else:
            self.user_messages.append(user_response)

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
        return f'{self.word}\n\n{self.interface["Example"][self.uilang]}: {self.assistant_responses[0]["example"]}'

    async def get_next_user_message(self, user_response: Optional[str]):
        lang_tr = self.interface[self.lang][self.uilang]
        if user_response is None:
            # Если перевод и пример уже есть в базе данных — используем их напрямую без LLM!
            has_data = (self.translation and not pd.isna(self.translation) and
                        self.example_sentence and not pd.isna(self.example_sentence) and
                        self.example_translation and not pd.isna(self.example_translation))
            if has_data:
                self.assistant_responses.append(dict(
                    example=str(self.example_sentence),
                    translation_example=str(self.example_translation),
                    translation_word=str(self.translation)
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
                
                self.assistant_responses.append(dict(example=assistant_response.example, translation_example=assistant_response.translation_of_example,
                                                     translation_word=assistant_response.translation_of_word))

            message_template = self.templates.get_template(self.uilang, self.lang, 'flashcard_user_message_1')
            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            message = template.render(lang=lang_tr, lang_ui=self.uilang, word=self.assistant_responses[0]['translation_word'], example=self.assistant_responses[0]['translation_example'])
            quality = None
            
        else:
            self.user_messages.append(user_response)

            message_template = self.templates.get_template(self.uilang, self.lang, 'flashcard_query_2')

            template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            query = template.render(lang=self.interface[self.lang][self.uilang], user_response=user_response,
                                    word_translation=self.assistant_responses[-1]['translation_word'], correct_answer=self.word)

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
            context_translation = self.assistant_responses[-1]['example']
            message = template.render(score=assistant_response.translation_score,
                                justification=assistant_response.score_justification,
                                correct_answer=correct_answer,
                                context_translation=context_translation)
            quality = assistant_response.translation_score

        return message, quality
