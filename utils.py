import asyncio
import json
import os
from pathlib import Path
import random
import re
import time
import warnings
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai

# Карта голосов edge-tts для разных языков
EDGE_TTS_VOICES = {
    'french': 'fr-FR-DeniseNeural',
    'spanish': 'es-ES-ElviraNeural',
    'italian': 'it-IT-ElsaNeural',
    'russian': 'ru-RU-SvetlanaNeural',
    'english': 'en-US-AriaNeural',
    'uzbek': 'uz-UZ-MadinaNeural',
    'japanese': 'ja-JP-NanamiNeural',
}


def parse_json_safely(content: str, validation_cls=None):
    """
    Безопасно парсит JSON от LLM и валидирует через Pydantic.
    Устойчив к неэкранированным кавычкам, переводам строк внутри значений,
    незакрытым скобкам (truncated JSON) и маркдаун-обёрткам.
    """
    if not content or not str(content).strip():
        raise ValueError("Пустой ответ от модели (empty content).")

    text = str(content).strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    # Очистка от непечатных ASCII control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # Поиск первой и последней фигурной/квадратной скобки
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not match:
        raise ValueError(f"В ответе модели не найден JSON-объект. Получено: {repr(text[:150])}")
    candidate = match.group(0)

    # Попытка 1: стандартный json.loads с strict=False
    try:
        data = json.loads(candidate, strict=False)
    except Exception:
        # Попытка 2: починка неэкранированных внутренних кавычек и закрытие незавершённых строк/скобок
        pattern_closed = re.compile(r'^(\s*"[a-zA-Z0-9_]+"\s*:\s*")(.*)(",?\s*)$')
        pattern_unclosed = re.compile(r'^(\s*"[a-zA-Z0-9_]+"\s*:\s*")(.*)$')
        fixed_lines = []
        for line in candidate.splitlines():
            m = pattern_closed.match(line)
            if m:
                prefix, val_content, suffix = m.groups()
                val_content = val_content.replace('\\"', '"').replace('"', '\\"')
                fixed_lines.append(prefix + val_content + suffix)
            else:
                m2 = pattern_unclosed.match(line)
                if m2 and not line.rstrip().endswith(('}', ']', '{', '[')):
                    prefix, val_content = m2.groups()
                    val_content = val_content.replace('\\"', '"').replace('"', '\\"')
                    fixed_lines.append(prefix + val_content + '"')
                else:
                    fixed_lines.append(line)
        repaired = '\n'.join(fixed_lines)

        # Балансировка незакрытых кавычек и фигурных/квадратных скобок
        in_str = False
        esc = False
        stack = []
        for ch in repaired:
            if ch == '"' and not esc:
                in_str = not in_str
            elif not in_str:
                if ch in ('{', '['):
                    stack.append('}' if ch == '{' else ']')
                elif ch in ('}', ']'):
                    if stack and stack[-1] == ch:
                        stack.pop()
            if ch == '\\' and not esc:
                esc = True
            else:
                esc = False

        if in_str:
            repaired += '"'
        while stack:
            repaired += stack.pop()

        try:
            data = json.loads(repaired, strict=False)
        except Exception:
            # Попытка 3: экранирование переводов строк внутри строк
            def escape_newlines(s: str) -> str:
                in_string = False
                escaped = False
                res = []
                for ch in s:
                    if ch == '"' and not escaped:
                        in_string = not in_string
                        res.append(ch)
                    elif in_string and ch == '\n':
                        res.append('\\n')
                    elif in_string and ch == '\r':
                        res.append('\\r')
                    elif in_string and ch == '\t':
                        res.append('\\t')
                    else:
                        res.append(ch)
                    if ch == '\\' and not escaped:
                        escaped = True
                    else:
                        escaped = False
                return ''.join(res)

            data = json.loads(escape_newlines(repaired), strict=False)

    if validation_cls is not None:
        # Gemini иногда возвращает массив объектов вместо объекта-обёртки.
        # Если тип списка, но схема ожидает объект с полем 'words', — автоматически адаптируем.
        if isinstance(data, list):
            # Пробуем достать поле 'words' из первого элемента (старый вариант)
            # Или оборачиваем список как {'words': [...], 'deck_theme': ...}
            if data and isinstance(data[0], dict):
                first = data[0]
                deck_theme = first.get('deck_theme', '')
                # Если каждый элемент содержит deck_theme на верхнем уровне — это список слов
                if 'word' in first or 'translation' in first:
                    data = {'deck_theme': deck_theme, 'words': data}
                else:
                    data = data[0]
        return validation_cls.model_validate(data)
    return data


async def get_assistant_response(interface, query, uilang, model_base, model_substitute,
                                  response_format=None, validation_cls=None):
    """Отправляет запрос в Gemini API и возвращает ответ с автоматическим повтором при ошибках."""

    api_key = os.getenv('GEMINI_API_KEY')
    if api_key is None:
        raise ValueError('Переменная окружения GEMINI_API_KEY не задана.')

    genai.configure(api_key=api_key)

    system_prompt = interface["You are a great language teacher"][uilang]
    model_name = model_base or os.getenv('MODEL_BASE', 'gemini-3.5-flash-lite')
    max_attempts = 4
    nattempts = 0
    last_error = None

    # Лимит токенов 4096 и температура 0.7 (предотвращает срабатывание фильтра цитирования RECITATION)
    gen_config_kwargs = {
        'max_output_tokens': 4096,
        'temperature': 0.7,
    }
    if validation_cls is not None or response_format is not None:
        gen_config_kwargs['response_mime_type'] = 'application/json'

    while nattempts < max_attempts:
        nattempts += 1
        try:
            # На повторных попытках повышаем температуру для максимальной вариативности
            if nattempts > 1:
                gen_config_kwargs['temperature'] = min(1.0, 0.7 + (nattempts - 1) * 0.12)

            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
            )
            generation_config = genai.GenerationConfig(**gen_config_kwargs)
            print(f'Отправляю запрос в Gemini ({model_name}, попытка {nattempts})...')
            response = await asyncio.to_thread(
                model.generate_content,
                query,
                generation_config=generation_config,
            )

            # Безопасное извлечение текста ответа
            content = None
            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                if finish_reason == 4:
                    raise ValueError("Сработал фильтр цитирования RECITATION. Повышаем вариативность и повторяем...")
                if finish_reason == 3:
                    raise ValueError("Сработал фильтр безопасности (SAFETY).")
                if finish_reason == 2:
                    raise ValueError("Достигнут лимит токенов (MAX_TOKENS).")
                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                    parts_text = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                    if parts_text:
                        content = "".join(parts_text).strip()

            if not content:
                try:
                    content = response.text.strip()
                except Exception:
                    pass

            if not content or not str(content).strip():
                raise ValueError(f"Модель {model_name} вернула пустой ответ (finish_reason={finish_reason}).")

            if validation_cls is not None:
                validated_resp = parse_json_safely(content, validation_cls)
            else:
                validated_resp = content

            print('Готово.')
            return validated_resp
        except Exception as e:
            last_error = e
            print(f'Ошибка обработки ответа Gemini (попытка {nattempts}): {e}')
            if nattempts < max_attempts:
                await asyncio.sleep(nattempts * 1.5)
                if nattempts == 2:
                    model_name = model_substitute or os.getenv('MODEL_SUBSTITUTE', 'gemini-3.5-flash-lite')
                elif nattempts >= 3:
                    # Гарантированный надежный откат к gemini-3.6-flash
                    model_name = 'gemini-3.6-flash'

    raise ValueError(f'Модель не смогла дать валидный ответ после {max_attempts} попыток. Последняя ошибка: {last_error}')



async def get_audio(query: str, lang: str, file_path: str) -> None:
    """Генерирует аудио с помощью edge-tts (нейросетевые голоса Microsoft)."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError('Установите edge-tts: pip install edge-tts')

    voice = EDGE_TTS_VOICES.get(lang.lower(), 'fr-FR-DeniseNeural')
    communicate = edge_tts.Communicate(query, voice)
    await communicate.save(file_path)
    print(f'Аудио сохранено: {file_path} (голос: {voice})')


def download_file(aurl, fpath):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        ),
        "Referer": "https://aiquickdraw.com/",
    }

    r = requests.get(aurl, headers=headers, stream=True)
    r.raise_for_status()

    with open(fpath, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


def generate_song_of_the_day(word, lang, dirpath):
    print(f'Generating song of the day')
    BASE_URL = "https://api.sunoapi.org"
    url = f"{BASE_URL}/api/v1/generate"

    with open('resources/song_genres.json', 'r') as fp:
        genres = json.loads(fp.read())

    genre_prompt = ''
    if genres:
        genre = random.sample(genres, 1)
        genre_prompt = f" Genre: {genre[0]}."
    payload = {
        "customMode": False,
        "model": "V4_5",
        "prompt": f"Create a song in {lang} that uses in its lyrics the word '{word}'.{genre_prompt}",
        "instrumental": False,
        "callBackUrl": "test.url"
    }
    suno_api_key = os.getenv('SUNO_API_KEY')
    headers = {
        "Authorization": f"Bearer {suno_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    task_id = response.json()['data']["taskId"]

    print(f'Sent generation request, waiting...')

    url = f"{BASE_URL}/api/v1/generate/record-info"
    n_iter = 1
    while True:
        response = requests.get(url, headers=headers, params={"taskId": task_id})
        response.raise_for_status()

        data = response.json()
        status = data['data']['status']

        if status == "SUCCESS":
            print(f'Song generation complete')

            audio_url_field = 'audioUrl'
            audio_data = [{'audio_url': d[audio_url_field], 'audio_file_path': dirpath / f'f{didx}.mp3'}
                          for didx, d in enumerate(data['data']['response']['sunoData'])]
            data = {'img_path': dirpath / f'img.jpg', 'img_url': data['data']['response']['sunoData'][0]['imageUrl'],
                    'lyrics': data['data']['response']['sunoData'][0]['prompt'], 'title': data['data']['response']['sunoData'][0]['title'], 'audio_data': audio_data}

            download_file(data['img_url'], str(data['img_path']))

            for aurl in audio_data:
                download_file(aurl['audio_url'], str(aurl['audio_file_path']))

            return data
        elif status == "PENDING" or 'SUCCESS' in status:
            print("Still generating...")
            time.sleep(5 * n_iter)
            n_iter += 1
            if n_iter > 20:
                raise Exception('Song generation took too long.')
        elif "FAIL" in status:
            raise Exception(status)