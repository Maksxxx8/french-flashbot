import copy
import itertools
import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo


DEFAULT_CONFIG = {
    "language": "french",
    "ui_language": "russian",
    "timezone": "Europe/Moscow",
    "level": "A1",
    "max_tokens": 800,
    "show_words_due": True,
    "n_flashcards": 5,
    "exercise_types": ["words"],
    "schedule": {
        "words": {
            "weekday": {
                "09:00": "learn",
                "12:00": "test",
                "15:00": "learn",
                "19:00": "test",
                "21:00": "test_translation"
            },
            "weekend": {
                "10:00": "learn",
                "13:00": "test",
                "16:00": "learn",
                "20:00": "test",
                "22:00": "test_translation"
            }
        }
    }
}


class UserConfig:
    def __init__(self, path):
        self.data_path = path
        self._lock = threading.Lock()

        try:
            with open(self.data_path, 'r', encoding='utf-8') as fp:
                self._user_data_orig = json.loads(fp.read())
        except Exception:
            self._user_data_orig = {"default": copy.deepcopy(DEFAULT_CONFIG)}

        self._default_template = copy.deepcopy(DEFAULT_CONFIG)
        self._user_data = {}

        # Извлекаем шаблон и реальные chat_id
        for k, v in self._user_data_orig.items():
            try:
                chat_id_int = int(k)
                self._user_data[chat_id_int] = copy.deepcopy(v)
            except (ValueError, TypeError):
                # Нечисловой ключ (например 'default' или '<ВСТАВЬ_СВОЙ_CHAT_ID>')
                self._default_template = copy.deepcopy(v)

        # Парсим расписание для всех существующих пользователей
        for chat_id in list(self._user_data.keys()):
            self._init_user_schedule(chat_id)

    def _init_user_schedule(self, chat_id):
        tz_str = self._user_data[chat_id].get("timezone", "Europe/Moscow")
        try:
            user_tz = ZoneInfo(tz_str)
        except Exception:
            user_tz = ZoneInfo("Europe/Moscow")

        if 'schedule' not in self._user_data[chat_id]:
            self._user_data[chat_id]['schedule'] = {}

        for exercise, day in itertools.product(['words'], ['weekday', 'weekend']):
            if exercise not in self._user_data[chat_id]['schedule']:
                self._user_data[chat_id]['schedule'][exercise] = {}

            if day in self._user_data[chat_id]['schedule'][exercise]:
                sched = self._user_data[chat_id]['schedule'][exercise][day]
                parsed = {}
                for x, data in sched.items():
                    if isinstance(x, str):
                        try:
                            t = datetime.strptime(x, '%H:%M').time().replace(tzinfo=user_tz)
                            parsed[t] = data
                        except Exception:
                            pass
                    else:
                        parsed[x] = data
                self._user_data[chat_id]['schedule'][exercise][day] = parsed
            else:
                self._user_data[chat_id]['schedule'][exercise][day] = {}

    def get_all_user_data(self):
        self._lock.acquire()
        ud_cpy = copy.deepcopy(self._user_data)
        self._lock.release()
        return ud_cpy

    def get_all_chat_ids(self):
        self._lock.acquire()
        chat_ids = list(self._user_data.keys())
        self._lock.release()
        return chat_ids

    def get_user_data(self, chat_id):
        self._lock.acquire()
        chat_id = int(chat_id)
        if chat_id not in self._user_data:
            # Автоматически регистрируем нового пользователя по шаблону
            self._user_data[chat_id] = copy.deepcopy(self._default_template)
            self._init_user_schedule(chat_id)
            self._user_data_orig[str(chat_id)] = copy.deepcopy(self._default_template)
            try:
                with open(self.data_path, 'w', encoding='utf-8') as fp:
                    json.dump(self._user_data_orig, fp, indent=2, ensure_ascii=False)
                print(f"Пользователь {chat_id} автоматически зарегистрирован в {self.data_path}")
            except Exception as e:
                print(f"Предупреждение: не удалось сохранить {self.data_path}: {e}")

        ud_cpy = copy.deepcopy(self._user_data[chat_id])
        self._lock.release()
        return ud_cpy

    def get_user_ui_lang(self, chat_id):
        self._lock.acquire()
        chat_id = int(chat_id)
        if chat_id in self._user_data and 'ui_language' in self._user_data[chat_id]:
            lang = self._user_data[chat_id]['ui_language']
        else:
            lang = self._default_template.get('ui_language', 'russian')
        self._lock.release()
        return lang

    def release_lock(self):
        if self._lock.locked():
            self._lock.release()