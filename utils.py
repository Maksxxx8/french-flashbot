import asyncio
import json
import os
from pathlib import Path
import random
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


async def get_assistant_response(interface, query, uilang, model_base, model_substitute,
                                  response_format=None, validation_cls=None):
    """Отправляет запрос в Gemini API и возвращает ответ."""

    api_key = os.getenv('GEMINI_API_KEY')
    if api_key is None:
        raise ValueError('Переменная окружения GEMINI_API_KEY не задана.')

    genai.configure(api_key=api_key)

    system_prompt = interface["You are a great language teacher"][uilang]
    model_name = model_base or os.getenv('MODEL_BASE', 'gemini-3.6-flash')
    max_attempts = 3
    nattempts = 0
    response = None

    # Настройка генерации — структурированный JSON если передана схема
    gen_config_kwargs = {'max_output_tokens': 1000}
    if validation_cls is not None:
        gen_config_kwargs['response_mime_type'] = 'application/json'
        gen_config_kwargs['response_schema'] = validation_cls

    while nattempts < max_attempts:
        nattempts += 1
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
            )
            generation_config = genai.GenerationConfig(**gen_config_kwargs)
            print(f'Отправляю запрос в Gemini ({model_name})...')
            response = await asyncio.to_thread(
                model.generate_content,
                query,
                generation_config=generation_config,
            )
            break
        except Exception as e:
            print(f'Ошибка Gemini (попытка {nattempts}): {e}')
            await asyncio.sleep(nattempts * 2)
            if nattempts == max_attempts - 1:
                model_name = model_substitute or os.getenv('MODEL_SUBSTITUTE', 'gemini-3.6-flash')

    if response is None:
        raise ValueError('Модель не смогла ответить после нескольких попыток.')

    print('Готово.')
    content = response.text.strip()
    if content.startswith('```'):
        lines = content.splitlines()
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        content = '\n'.join(lines).strip()

    if validation_cls is not None:
        validated_resp = validation_cls.model_validate_json(content)
    else:
        validated_resp = content

    return validated_resp



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