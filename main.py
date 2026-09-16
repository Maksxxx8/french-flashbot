import asyncio
import json
import os.path
from datetime import datetime, timedelta
from pathlib import Path
import signal
from zoneinfo import ZoneInfo

import jinja2
from flask import Flask
import requests

from telegram import Update, BotCommand
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackQueryHandler, ContextTypes
import telegramify_markdown
import pandas as pd

from decks_db import DecksDB
from exercise import Exercise
from learning_plan import LearningPlan
from templates import Templates
from utils import get_audio, generate_song_of_the_day
from words_progress_db import WordsProgressDB
from user_config import UserConfig
from words_db import WordsDB
from words_exercise import FlashcardExercise, WordsExerciseLearn, WordsExerciseTest, normalize_french_text
from running_activities import RunningActivities
import dotenv
from tempfile import TemporaryDirectory

import logging

dotenv.load_dotenv()

# Disable optional logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)


app = Flask(__name__)

BOT_TOKEN_ENG = os.getenv('BOT_TOKEN_ENG')
BOT_TOKEN_RU = os.getenv('BOT_TOKEN_RU')

BOT_TOKENS = [BOT_TOKEN_ENG]
BOT_LANGS = ['russian']
if BOT_TOKEN_RU is not None:
    BOT_TOKENS.append(BOT_TOKEN_RU)
    BOT_LANGS.append('russian')


async def handle_new_exercise(bot, chat_id, exercise):
    try:
        user_data = user_config.get_user_data(chat_id)
        lang = user_data['language']
        uilang = lang_map[bot.token]

        running_activities.add_activity(chat_id, exercise)
        if isinstance(exercise, WordsExerciseLearn):
            lp.process_response(chat_id, exercise, quality=None)
            words_progress_db.save_progress()

        try:
            if not isinstance(exercise, WordsExerciseLearn) and "show_words_due" in user_data.keys() and user_data['show_words_due']:
                words_due = lp.get_due_today(chat_id, lang)

                message_template = templates.get_template(uilang, lang, 'words_due')
                template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
                due_message = template.render(n_words=len(words_due)).strip()
                if len(words_due) > 0:
                    due_message = f'{due_message}\n\n---------------\n\n'
            else:
                due_message = ''

            message, _ = await exercise.get_next_user_message(user_response=None)

            message = f'{due_message}{message}'

            if isinstance(exercise, WordsExerciseLearn):
                buttons = ['Next']
            elif isinstance(exercise, WordsExerciseTest):
                buttons = ['Easier', 'Harder', 'Hint', 'Correct answer', 'Answer audio']
            elif isinstance(exercise, FlashcardExercise):
                buttons = ['Correct answer']
        except Exception as e:
            message = f'{interface["Error"][uilang]}: {e}'
            buttons = None

        if isinstance(exercise, WordsExerciseLearn):
            # 1. Отправляем текст карточки
            await tel_send_message(bot, chat_id, message, buttons=None)

            # 2. Сразу отправляем голосовое сообщение с кнопкой 'Next'
            pre_audio = getattr(exercise, 'audio_path', None)
            audio_sent = False
            if pre_audio and os.path.exists(pre_audio):
                try:
                    await tel_send_audio(bot, chat_id, pre_audio, as_voice=True, buttons=['Next'])
                    audio_sent = True
                except Exception as e:
                    print(f'Error sending cached voice: {e}')

            if not audio_sent:
                file_path = f'{chat_id}_{exercise.uid}.mp3'
                try:
                    await get_audio(exercise.word, exercise.lang, file_path)
                    await tel_send_audio(bot, chat_id, file_path, as_voice=True, buttons=['Next'])
                    audio_sent = True
                except Exception as e:
                    print(f'Error generating/sending voice: {e}')
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)

            if not audio_sent:
                await tel_send_message(bot, chat_id, "👉", buttons=['Next'])
        else:
            await tel_send_message(bot, chat_id, message, buttons=buttons)
    except Exception as e:
        if chat_id in running_activities.chat_ids: running_activities.pop_all(chat_id)
        release_all_locks()
        print(e)
        await tel_send_message(bot, chat_id, interface['Something went terribly wrong, please try again or notify the admin'][uilang])


async def ping_user(bot, chat_id, lang, exercise_type, exercise_data):
    uilang = lang_map[bot.token]
    if exercise_type == 'words':
        exercise = await lp.get_next_words_exercise(chat_id, lang, mode=exercise_data)
    else:
        raise ValueError(f'Unknown exercise type {exercise_type}')

    if exercise is None:
        if exercise_data in ['test', 'test_flashcard', 'test_translation']:
            print(f'No words due for review for {chat_id} at this time. Skipping ping.')
        else:
            await tel_send_message(bot, chat_id, interface['Could not create an exercise, will try again later'][uilang])
            print(f'Could not create an exercise {exercise_type} for data {exercise_data}.')
    else:
        await handle_new_exercise(bot, chat_id, exercise)


async def tel_send_audio(bot, chat_id, audio_file_path, title='audio.mp3', as_voice=True, buttons=None):
    uilang = lang_map.get(bot.token, 'russian')
    payload = {
        'chat_id': str(chat_id)
    }
    if buttons is not None:
        buttons_list = []
        for button_text in buttons:
            buttons_list.append(
                {
                    "text": interface[button_text][uilang] if button_text in interface.keys() else button_text,
                    "callback_data": button_text
                }
            )
        reply_markup = {
            "inline_keyboard": [[b] for b in buttons_list]
        }
        payload['reply_markup'] = json.dumps(reply_markup)

    with open(audio_file_path, 'rb') as audio_file:
        if as_voice:
            files = {
                'voice': (title, audio_file.read(), 'audio/mpeg')
            }
            resp = requests.post(
                f"https://api.telegram.org/bot{bot.token}/sendVoice",
                data=payload,
                files=files)
            if not resp.ok:
                print(f'sendVoice error: {resp.status_code} {resp.text}')
                raise RuntimeError(f'sendVoice failed: {resp.text}')
        else:
            payload['title'] = title
            payload['parse_mode'] = 'HTML'
            files = {
                'audio': audio_file.read(),
            }
            resp = requests.post(
                f"https://api.telegram.org/bot{bot.token}/sendAudio",
                data=payload,
                files=files)
            if not resp.ok:
                print(f'sendAudio error: {resp.status_code} {resp.text}')
                raise RuntimeError(f'sendAudio failed: {resp.text}')



async def tel_send_image(bot, chat_id, image_file_path, caption):
    with open(image_file_path, 'rb') as img_file:
        payload = {
            'chat_id': str(chat_id),
            'caption': caption,
            'parse_mode': 'HTML'
        }
        files = {
            'photo': img_file.read(),
        }

        resp = requests.post(
            f"https://api.telegram.org/bot{bot.token}/sendPhoto",
            data=payload,
            files=files)
        resp.json()

async def tel_send_message(bot, chat_id, text, buttons=None, parse_mode="MarkdownV2"):

    # determine max line length to know if to display buttons on separate lines or on the same one
    lines = text.split('\n')
    max_len = max([len(line) for line in lines])
    uilang = lang_map[bot.token]
    reply_markup = None
    if buttons is not None:
        buttons_list = []
        for button_text in buttons:
            buttons_list.append(
                {
                    "text": interface[button_text][uilang] if button_text in interface.keys() else button_text,
                    "callback_data": button_text
                }
            )

        buttons_to_send = [[button] for button in buttons_list] if max_len < 74 else [buttons_list]
        reply_markup = {
            "inline_keyboard": buttons_to_send
        }

    converted = telegramify_markdown.markdownify(text)
    await bot.send_message(chat_id, converted, reply_markup=reply_markup,  parse_mode=parse_mode)


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    bot = context._application.bot
    uilang = lang_map[bot.token]
    raw_cmd = update.message.text[1:] if update.message and update.message.text else ''
    command = raw_cmd.split()[0].split('@')[0].lower()
    chat_id = update.message.chat_id

    try:
        if command == 'start':
            await handle_start(update, context)
        elif command == 'help':
            await handle_help(update, context)
        elif command == 'add_word':
            await handle_add_word(update, context)
        elif command == 'next_new':
            await handle_next_new(update, context)
        elif command == 'next_test':
            await handle_next_test(update, context)
        elif command in ['stop', 'notifications_off', 'unsubscribe']:
            await handle_notifications_off(update, context)
        elif command in ['notifications_on', 'subscribe']:
            await handle_notifications_on(update, context)
        elif command in ['reset', 'reset_progress']:
            await handle_reset(update, context)
        elif command == 'stats':
            await handle_stats(update, context)
        elif command == 'words':
            await handle_words(update, context)
        elif command == 'replay' or command.startswith('replay_'):
            await handle_replay(update, context, raw_cmd)
    except Exception as e:
        if chat_id in running_activities.chat_ids: running_activities.pop_all(chat_id)
        release_all_locks()
        print(e)
        await tel_send_message(bot, chat_id, interface['Something went terribly wrong, please try again or notify the admin'][uilang])




async def handle_stats(update, context):
    chat_id = update.message.chat_id
    bot = context._application.bot
    lang = user_config.get_user_data(chat_id)['language']

    stats = lp.get_user_stats(chat_id, lang)

    if stats['total'] == 0:
        await tel_send_message(bot, chat_id, "ℹ️ Статистика пуста. У вас нет добавленных слов.")
        return

    lines = [
        f"📊 **Ваша статистика ({lang}):**",
        "",
        f"📚 Всего слов в словаре: *{stats['total']}*",
        f"🟦 Новых (неизученных): *{stats['unseen']}*",
        f"🔄 В процессе изучения: *{stats['learning']}*",
        f"🟩 Выучено (интервал ≥21 дн.): *{stats['learned']}*",
        "",
        "📖 Для просмотра изученных слов нажмите /words"
    ]
    message = "\n".join(lines)
    await tel_send_message(bot, chat_id, message)


async def handle_replay(update, context, command):
    chat_id = update.message.chat_id
    bot = context._application.bot
    lang = user_config.get_user_data(chat_id)['language']
    uilang = lang_map[bot.token]
    
    try:
        raw_text = command.strip().lstrip('/')
        clean_first = raw_text.split('@')[0]
        if '_' in clean_first:
            word_id = int(clean_first.split('_')[1])
        else:
            word_id = int(clean_first.split()[1])
    except Exception as e:
        print(f"Error parsing replay command '{command}': {e}")
        return
        
    exercise = await lp.get_words_exercise_by_id(chat_id, lang, word_id, mode='learn')
    if exercise is None:
        await tel_send_message(bot, chat_id, "Слово не найдено.")
        return

    # 1. Отправляем текст карточки
    try:
        message, _ = await exercise.get_next_user_message(user_response=None)
        await tel_send_message(bot, chat_id, message)
    except Exception as e:
        print(f"Error rendering replay message for word_id={word_id}: {e}")
        await tel_send_message(bot, chat_id, f"🇫🇷 **{exercise.word}**")

    # 2. Отправляем озвучку (голосовое сообщение)
    pre_audio = getattr(exercise, 'audio_path', None)
    audio_sent = False
    if pre_audio and os.path.exists(pre_audio):
        try:
            await tel_send_audio(bot, chat_id, pre_audio, as_voice=True)
            audio_sent = True
        except Exception as e:
            print(f'Error sending cached voice for word {exercise.word}: {e}')

    if not audio_sent:
        file_path = f'{chat_id}_{exercise.uid}.mp3'
        try:
            await get_audio(exercise.word, exercise.lang, file_path)
            await tel_send_audio(bot, chat_id, file_path, as_voice=True)
        except Exception as e:
            print(f'Error generating audio for {exercise.word}: {e}')
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


async def handle_words(update, context):
    chat_id = update.message.chat_id
    bot = context._application.bot
    lang = user_config.get_user_data(chat_id)['language']

    learned_words = lp.get_user_learned_words(chat_id, lang, limit=20)

    if not learned_words:
        await tel_send_message(bot, chat_id, "ℹ️ Вы пока не выучили ни одного слова. Используйте /next_new для начала обучения.")
        return

    msg_lines = [
        "📖 **Изученные слова (Топ-20 по интервалу повторения):**",
        ""
    ]
    for row in learned_words:
        msg_lines.append(f"• 🇫🇷 **{row['word']}** (интервал: {row['interval']} дн.) — /replay_{row['word_id']}")
    
    msg_lines.extend([
        "",
        "💡 Нажмите /replay_<номер> рядом со словом, чтобы прослушать произношение и карточку."
    ])
    await tel_send_message(bot, chat_id, "\n".join(msg_lines))


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    user_data = user_config.get_user_data(chat_id)
    is_notif = user_config.is_notifications_enabled(chat_id)
    notif_status = "🔔 Включены (5 раз в день)" if is_notif else "🔕 Отключены"

    welcome_msg = (
        f"🇫🇷 *Bonjour !* Я бот для изучения французского языка для поездки в Париж (уровень A1 → A2).\n\n"
        f"Уведомления по расписанию: *{notif_status}*\n\n"
        f"📌 *Основные команды:*\n"
        f"• /next_new — Новое слово (с артиклем, транскрипцией, примером и сразу озвучкой)\n"
        f"• /next_test — Проверить выученные слова (тест/флэшкарты)\n"
        f"• /stats — Статистика изучения и прогресс по словам\n"
        f"• /words — Список изученных слов с возможностью прослушать карточки\n"
        f"• /add_word — Добавить свое слово для изучения\n"
        f"• /help — Подробная справка по всем командам и кнопкам\n\n"
        f"⚙️ *Уведомления и подписка:*\n"
        f"• /stop или /notifications_off — Отключить авторассылку по расписанию\n"
        f"• /notifications_on — Включить авторассылку по расписанию\n\n"
        f"Нажмите /next_new чтобы начать!"
    ) if uilang == 'russian' else (
        f"🇫🇷 *Bonjour!* I am your French vocabulary bot for your trip to Paris (A1 → A2).\n\n"
        f"Notifications: *{notif_status}*\n\n"
        f"Commands:\n"
        f"• /next_new — Next new word (with audio & examples)\n"
        f"• /next_test — Test learned words\n"
        f"• /stats — View vocabulary statistics\n"
        f"• /words — List studied words with replay audio\n"
        f"• /add_word — Add word manually\n"
        f"• /stop — Disable notifications\n"
        f"• /notifications_on — Enable notifications\n"
        f"• /help — Detailed help guide\n\n"
        f"Press /next_new to start!"
    )
    await tel_send_message(bot, chat_id, welcome_msg)

    # Проверяем и запускаем фоновую предгенерацию буфера слов
    lang = user_data.get('language', 'french')
    asyncio.create_task(lp.ensure_word_buffer(chat_id, lang))


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    is_notif = user_config.is_notifications_enabled(chat_id)
    notif_status = "🔔 Включены (5 раз в день)" if is_notif else "🔕 Отключены"

    help_msg = (
        f"📖 *СПРАВКА ПО БОТУ И КОМАНДАМ*\n\n"
        f"🎯 *Цель бота:* изучение ключевых французских слов и выражений для комфортной поездки в Париж (кафе, отель, метро, вокзал, улицы, музеи) на уровне A1.\n\n"
        f"Статус уведомлений для вас: *{notif_status}*\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *КОМАНДЫ БОТА:*\n"
        f"• /next_new — Показать следующее новое слово. Бот присылает карточку с переводом, транскрипцией, контекстным примером и сразу голосовым сообщением с правильным произношением.\n"
        f"• /next_test — Проверить изученные слова. Тестируются только те слова, которые вы уже учили, строго в рамках уровня A1 (без сложной грамматики).\n"
        f"• /stats — Посмотреть статистику (всего слов, в процессе, выучено).\n"
        f"• /words — Посмотреть список изученных слов с озвучкой карточек.\n"
        f"• /add_word — Добавить свое слово для изучения (введите слово после команды).\n"
        f"• /stop (или /notifications_off) — Отключить авторассылку по расписанию.\n"
        f"• /notifications_on (или /subscribe) — Включить напоминания по расписанию (09:00, 12:00, 15:00, 19:00, 21:00).\n"
        f"• /reset — Сбросить прогресс изученных слов (начать заново с чистого листа).\n"
        f"• /help — Открыть это руководство.\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔘 *КНОПКИ В УПРАЖНЕНИЯХ:*\n\n"
        f"🟢 *При изучении новых слов (/next_new):*\n"
        f"• *Next* (Далее) — перейти к следующему слову.\n\n"
        f"🔵 *При проверке и тестах (/next_test):*\n"
        f"• *Easier* (Легче) — уменьшить сложность тестовой фразы (сделать предложение короче и проще).\n"
        f"• *Harder* (Сложнее) — увеличить сложность тестовой фразы (в рамках уровня A1).\n"
        f"• *Hint* (Подсказка) — показать подсказку (требует написать слово 5 раз для закрепления).\n"
        f"• *Correct answer* (Правильный ответ) — сразу показать правильный ответ с переводом.\n"
        f"• *Answer audio* (Озвучить ответ) — прослушать произношение правильного ответа.\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *Как отвечать в тестах?*\n"
        f"Просто напишите перевод в чат обычным сообщением. Бот оценит ответ словами: *плохо*, *хорошо* (допускается опечатка в 1 букву или артикль) или *молодец*!"
    ) if uilang == 'russian' else (
        f"📖 *BOT GUIDE & COMMANDS*\n\n"
        f"Notifications: *{notif_status}*\n\n"
        f"Commands:\n"
        f"• /next_new — Next new word\n"
        f"• /next_test — Test learned words\n"
        f"• /add_word — Add word manually\n"
        f"• /stop or /notifications_off — Disable notifications\n"
        f"• /notifications_on — Enable notifications\n"
        f"• /help — This help message\n\n"
        f"Buttons:\n"
        f"• Next — Continue\n"
        f"• Easier / Harder — Adjust sentence difficulty (1-5)\n"
        f"• Hint — Show hint\n"
        f"• Correct answer — Show solution\n"
        f"• Answer audio — Audio of answer\n"
    )
    await tel_send_message(bot, chat_id, help_msg)


async def handle_notifications_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    user_config.set_notifications(chat_id, enabled=False)

    msg = (
        "🔕 *Уведомления по расписанию отключены.*\n\n"
        "Бот больше не будет автоматически присылать слова и тесты в течение дня.\n\n"
        "Вы всегда можете заниматься в любое удобное время самостоятельно:\n"
        "• /next_new — Учить новое слово\n"
        "• /next_test — Проверить выученные слова\n\n"
        "Чтобы снова включить автоматические напоминания по расписанию, отправьте /notifications_on или /subscribe."
    ) if uilang == 'russian' else (
        "🔕 *Notifications disabled.*\n\n"
        "Scheduled reminders have been turned off. You can still practice anytime manually via /next_new and /next_test.\n\n"
        "To re-enable scheduled reminders, send /notifications_on or /subscribe."
    )
    await tel_send_message(bot, chat_id, msg)


async def handle_notifications_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    user_config.set_notifications(chat_id, enabled=True)

    msg = (
        "🔔 *Уведомления по расписанию включены!*\n\n"
        "Бот будет присылать напоминания для изучения и повторения слов 5 раз в день:\n"
        "• Будни: 09:00, 12:00, 15:00, 19:00, 21:00\n"
        "• Выходные: 10:00, 13:00, 16:00, 20:00, 22:00\n\n"
        "Чтобы отключить напоминания в любой момент, отправьте /stop или /notifications_off."
    ) if uilang == 'russian' else (
        "🔔 *Notifications enabled!*\n\n"
        "You will receive reminders 5 times a day according to schedule.\n\n"
        "To disable, send /stop or /notifications_off."
    )
    await tel_send_message(bot, chat_id, msg)


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]

    words_progress_db.reset_user_progress(chat_id)
    if chat_id in running_activities.chat_ids:
        running_activities.pop_all(chat_id)

    msg = (
        "🔄 *Ваш прогресс изучения слов успешно сброшен!*\n\n"
        "Бот забыл все пройденные слова и повторения для вашего аккаунта.\n\n"
        "Вы можете начать изучение заново с чистого листа:\n"
        "Нажмите /next_new, чтобы получить первое слово!"
    ) if uilang == 'russian' else (
        "🔄 *Your vocabulary progress has been reset!*\n\n"
        "All learned words have been cleared. Press /next_new to start fresh!"
    )
    await tel_send_message(bot, chat_id, msg)


async def handle_add_word(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    running_activities.add_activity(chat_id, 'add_word')
    await tel_send_message(bot, chat_id, interface['Type the word that you would like to add'][uilang])


async def handle_next_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    lang = user_config.get_user_data(chat_id)['language']

    await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
    exercise = await lp.get_next_words_exercise(chat_id, lang, mode='test')
    if exercise is None:
        await tel_send_message(bot, chat_id, "🎉 *Все запланированные слова на сегодня уже повторены!*\n\nКаждое слово повторяется не чаще 1 раза в день. Следующие повторения откроются завтра по алгоритму интервальных повторений.\n\nЧтобы учить новые слова, отправьте команду /next_new")
        print(f'All words due for today are already reviewed for {chat_id}.')
    else:
        await handle_new_exercise(bot, chat_id, exercise)


async def handle_next_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    bot = context._application.bot
    uilang = lang_map[bot.token]
    lang = user_config.get_user_data(chat_id)['language']

    # Если буфер пуст — предупреждаем о генерации, иначе карточка откроется мгновенно!
    if lp.get_unseen_count(chat_id, lang) == 0:
        await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
    exercise = await lp.get_next_words_exercise(chat_id, lang, mode='learn')
    if exercise is None:
        await tel_send_message(bot, chat_id, interface['Could not create an exercise, please try again later'][uilang])
        print(f'Could not create an exercise "words" for data "learn".')
    else:
        await handle_new_exercise(bot, chat_id, exercise)
    
    if user_config.get_user_data(chat_id).get('song_of_the_day', False):

        with TemporaryDirectory() as fp:
            song_data = generate_song_of_the_day(word=exercise.word, lang=lang, dirpath=Path(fp))
            
            lyrics = '\n'.join([f'> {line}' for line in song_data['lyrics'].split('\n')])

            # await tel_send_message(bot, chat_id, mes)
            await tel_send_image(bot, chat_id, song_data['img_path'], exercise.word)
            await tel_send_message(bot, chat_id, lyrics)
            await tel_send_audio(bot, chat_id, song_data['audio_data'][0]['audio_file_path'], title=song_data['title'] + ' (1)', as_voice=False)
            await tel_send_audio(bot, chat_id, song_data['audio_data'][1]['audio_file_path'], title=song_data['title'] + ' (2)', as_voice=False)



async def handle_exercise_button_press(update, context, chat_id, lang, udata, exercise):
    try:
        print(f'Received a button: {udata}')
        bot = context._application.bot
        uilang = lang_map[bot.token]
        bot = context._application.bot

        if isinstance(exercise, WordsExerciseLearn) or isinstance(exercise, WordsExerciseTest) or isinstance(exercise, FlashcardExercise):
            if f'Hint' == udata:

                running_exercise = running_activities.pop_activity(chat_id)
                running_exercise.hint_clicked = True
                running_exercise.times_to_write_left = 5
                running_activities.add_activity(chat_id, running_exercise)

                if not exercise.is_responded:
                    lp.process_hint(chat_id, running_exercise)
                    words_progress_db.save_progress()

                message_template = templates.get_template(uilang, lang, 'hint_message')
                template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
                mes = template.render(word=exercise.word)
                mes += "\n\n⚠️ Вы подсмотрели подсказку. Напишите правильное слово 5 раз подряд, чтобы продолжить."
                await tel_send_message(bot, chat_id, mes)

            elif f'Correct answer' == udata:

                running_exercise = running_activities.pop_activity(chat_id)
                running_exercise.correct_answer_clicked = True
                running_activities.add_activity(chat_id, running_exercise)
                if not exercise.is_responded:
                    lp.process_correct_answer(chat_id, running_exercise)
                    words_progress_db.save_progress()

                await tel_send_message(bot, chat_id, f"🇫🇷 **{exercise.correct_answer()}**")
            # elif f'I know this word' == udata:
            #     lp.set_word_easy(chat_id, exercise.word_id)
            #     words_progress_db.save_progress()

            #     message_template = templates.get_template(uilang, lang, 'know_word_message')
            #     template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            #     mes = template.render(word=exercise.word)
            #     await tel_send_message(bot, chat_id, mes)

            #     progress_df = words_progress_db.get_progress_df()
            #     n_seen_words = progress_df[progress_df['chat_id'] == chat_id].shape[0]

            #     if n_seen_words % 10 == 0:

            #         message_template = templates.get_template(uilang, lang, 'congrats_learn_message')
            #         template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
            #         mes = template.render(n_seen_words=n_seen_words)
            #         await tel_send_message(bot, chat_id, mes)

            elif f'Answer audio' == udata:
                file_path = f'{chat_id}_{exercise.uid}.mp3'
                await get_audio(exercise.correct_answer(), exercise.lang, file_path)
                await tel_send_audio(bot, chat_id, file_path, as_voice=True)
                if os.path.exists(file_path):
                    os.remove(file_path)
            elif f'Pronounce' == udata:
                pre_audio = getattr(exercise, 'audio_path', None)
                if pre_audio and os.path.exists(pre_audio):
                    await tel_send_audio(bot, chat_id, pre_audio, as_voice=True)
                else:
                    file_path = f'{chat_id}_{exercise.uid}.mp3'
                    await get_audio(exercise.word, exercise.lang, file_path)
                    await tel_send_audio(bot, chat_id, file_path, as_voice=True)
                    if os.path.exists(file_path):
                        os.remove(file_path)
            elif f'Next' == udata:
                mode = 'learn' if isinstance(exercise, WordsExerciseLearn) else 'test'
                if mode == 'learn' and lp.get_unseen_count(chat_id, lang) == 0:
                    await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
                elif mode != 'learn':
                    await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
                exercise = await lp.get_next_words_exercise(chat_id, lang, mode)
                if exercise is None:
                    if mode == 'test':
                        await tel_send_message(bot, chat_id, "🎉 *Все запланированные слова на сегодня уже повторены!*\n\nКаждое слово повторяется не чаще 1 раза в день. Следующие повторения откроются завтра.\n\nЧтобы учить новые слова, нажмите /next_new.")
                    else:
                        await tel_send_message(bot, chat_id, "Все слова в текущем наборе пройдены. Генерирую новые слова, попробуйте через минуту /next_new.")
                    print(f'All words due for today are already reviewed or no more words for "{mode}".')
                else:
                    await handle_new_exercise(bot, chat_id, exercise)
            elif f'Easier' == udata:
                await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
                message = exercise.change_difficulty(easier=True)
                buttons = []
                if exercise.difficulty > 1:
                    buttons.append('Easier')
                if exercise.difficulty < 5:
                    buttons.append('Harder')
                buttons.extend(['Hint', 'Correct answer', 'Answer audio'])
                await tel_send_message(bot, chat_id, message, buttons=buttons)

            elif f'Harder' == udata:
                await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
                message = exercise.change_difficulty(easier=False)
                buttons = []
                if exercise.difficulty > 1:
                    buttons.append('Easier')
                if exercise.difficulty < 5:
                    buttons.append('Harder')
                buttons.extend(['Hint', 'Correct answer', 'Answer audio'])
                await tel_send_message(bot, chat_id, message, buttons=buttons)
            else:
                raise ValueError(f'Unknown callback data {udata}')
        else:
            raise ValueError(f'Unknown exercise type: {type(exercise)}.')
    except Exception as e:
        if chat_id in running_activities.chat_ids: running_activities.pop_all(chat_id)
        release_all_locks()
        print(e)
        await tel_send_message(bot, chat_id, interface['Something went terribly wrong, please try again or notify the admin'][uilang])


def execute_command_message(context, chat_id, lang, command, msg):
    bot = context._application.bot
    uilang = lang_map[bot.token]
    if command == 'add_word':
        words = msg.strip().split('\n')
        custom_deck_id = decks_db.get_custom_deck_id(str(chat_id), lang)
        for word in words:
            word_id = words_db.add_new_word(word, lang)
            decks_db.add_new_word(custom_deck_id, word_id)
        words_db.save_words_db()
        decks_db.save_decks_db()

        message_template = templates.get_template(uilang, lang, 'add_word_message')
        template = jinja2.Template(message_template, undefined=jinja2.StrictUndefined)
        user_msg = template.render(word=word)

    else:
        raise ValueError(f'Unexpected command {command}.')
    return user_msg


async def handle_inline_request(update, context):
    try:
        bot = context._application.bot
        uilang = lang_map[bot.token]
        chat_id = update.callback_query.from_user.id
        lang = user_config.get_user_data(chat_id)['language']

        data = update.callback_query.data
        button_text = data.split('_')[0]
        
        if button_text in exercise_buttons:

            activity = running_activities.current_activity(chat_id)
            if not isinstance(activity, Exercise) and activity is not None:
                # a command was pressed, but was not completed
                running_activities.pop_activity(chat_id)
            activity = running_activities.current_activity(chat_id)
            if activity is not None:
                await handle_exercise_button_press(update, context, chat_id, lang, data, activity)
            else:
                await tel_send_message(bot, chat_id, interface['Sorry, the exercise has been completed or is expired'][uilang])
        else:
            await tel_send_message(bot, chat_id, interface['Something went terribly wrong, please try again or notify the admin'][uilang])
            print(f'Unexpected inline request: {data}')

    except Exception as e:
        if chat_id in running_activities.chat_ids: running_activities.pop_all(chat_id)
        release_all_locks()
        print(e)
        await tel_send_message(bot, chat_id, interface['Something went terribly wrong, please try again or notify the admin'][uilang])


async def handle_request(update, context):
    try:
        chat_id = update.message.chat_id
        bot = context._application.bot
        uilang = lang_map[bot.token]
        msg = update.message.text
        print(f'{chat_id} message: {msg}')

        lang = user_config.get_user_data(chat_id)['language']

        current_activity = running_activities.current_activity(chat_id)

        if isinstance(current_activity, Exercise):
            # user responded to an exercise
            exercise = current_activity
            if not exercise.is_responded:
                
                if exercise.times_to_write_left > 0:
                    clean_msg = normalize_french_text(msg)
                    clean_target = normalize_french_text(exercise.word)
                    
                    if clean_msg == clean_target:
                        exercise.times_to_write_left -= 1
                        if exercise.times_to_write_left > 0:
                            await tel_send_message(bot, chat_id, f'Верно! Осталось написать еще {exercise.times_to_write_left} раз(а).')
                            return
                        else:
                            exercise.is_responded = True
                            lp.process_response(chat_id, exercise, quality=1)
                            words_progress_db.save_progress()
                            await tel_send_message(bot, chat_id, "🟩 **Оценка:** *плохо* (вы использовали подсказку)\n\nВы написали слово 5 раз. Можете переходить к следующему.", buttons=['Next'])
                            return
                    else:
                        await tel_send_message(bot, chat_id, f'Неверно. Вы написали "{msg}", а нужно "{exercise.word}". Осталось написать {exercise.times_to_write_left} раз(а).')
                        return

                exercise.is_responded = True

                await tel_send_message(bot, chat_id, f'{interface["Thinking"][uilang]}...')
                message, quality = await exercise.get_next_user_message(user_response=msg)
                
                buttons = None
                if isinstance(exercise, WordsExerciseLearn):
                    buttons = None
                elif isinstance(exercise, WordsExerciseTest) or isinstance(exercise, FlashcardExercise):
                    lp.process_response(chat_id, exercise, quality=quality)
                    words_progress_db.save_progress()
                    buttons = ['Next']

                await tel_send_message(bot, chat_id, message, buttons=buttons)
                words_progress_db.save_progress()
            else:
                await tel_send_message(bot, chat_id, f'{interface["The exercise has already been answered, this message will be ignored"][uilang]}: {msg}')
                print(f'The exercise has already been answered, this message will be ignored: {msg}')
            
        elif current_activity is not None:
            # handle an input for a command
            command = current_activity
            user_msg = execute_command_message(context, chat_id, lang, command, msg)
            await tel_send_message(bot, chat_id, user_msg)
            running_activities.pop_activity(chat_id)
        else:
            await tel_send_message(bot, chat_id, f'{interface["No exercises or commands are running, this message will be ignored"][uilang]}: {msg}')
            print(f'No running commands or exercises, ignore user message: {msg}')

    except Exception as e:
        if chat_id in running_activities.chat_ids: running_activities.pop_all(chat_id)
        release_all_locks()
        print(e)
        await tel_send_message(bot, chat_id, interface['Something went terribly wrong, please try again or notify the admin'][uilang])
    return


def release_all_locks():
    for shared_obj in shared_objs:
        shared_obj.release_lock()


async def ping_users(context):
    user_data = user_config.get_all_user_data()
    bot = context._application.bot
    uilang = lang_map[bot.token]
    is_weekend = (datetime.now().weekday() == 5 or datetime.now().weekday() == 6)
    schedule_col = 'weekend' if is_weekend else 'weekday'

    users_to_ping = []
    for chat_id, v in user_data.items():
        if v.get('ui_language') != uilang: continue
        if not v.get('notifications_enabled', True): continue
        if 'words' not in v.get('exercise_types', []): continue

        user_now = datetime.now(tz=ZoneInfo(user_data[chat_id]["timezone"]))
        ping_schedule = user_data[chat_id]['schedule']['words'][schedule_col]

        user_ping_times = [datetime.combine(user_now.date(), ptime) for ptime in ping_schedule.keys()]

        ping_schedule = [pexersize for uptime, pexersize in zip(user_ping_times, ping_schedule.values()) 
                            if abs(uptime - user_now) <= timedelta(minutes=1)]
        if len(ping_schedule) == 0:
            continue
        users_to_ping.append(dict(chat_id=chat_id, lang=user_data[chat_id]['language'], exercise=ping_schedule[0]))

    try:
        for user in users_to_ping:
            await ping_user(bot, user['chat_id'], user['lang'], 'words', user['exercise'])
    except Exception as e:
        if chat_id in running_activities.chat_ids: running_activities.pop_all(chat_id)
        release_all_locks()
        print(e)


def nearest_start_time(ping_interval=15 * 60):

    n_pings_per_hour = 60 * 60 // ping_interval
    ping_times = [ping_interval * (i + 1) // 60 for i in range(n_pings_per_hour)]
    if 60 not in ping_times:
        ping_times.append(60)

    now = datetime.now()
    starting_point_min = [ping_time for ping_time in ping_times if ping_time - now.minute >= 1][0]

    next_starting_point = now + timedelta(minutes=starting_point_min-now.minute)

    next_sec = (next_starting_point-datetime.now()).seconds
    if next_sec < 0: next_sec = 10
    return next_sec


async def run_apps(apps):

    stop_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        print('SIGINT or CTRL-C detected. Exiting gracefully')

        # store running exercises
        running_activities.backup()
    
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for app in apps:
        await app.initialize()
        await app.start()
        try:
            await app.bot.set_my_commands([
                BotCommand("next_new", "Новое слово (с озвучкой)"),
                BotCommand("next_test", "Тест / повторение выученного"),
                BotCommand("stats", "Статистика и прогресс"),
                BotCommand("words", "Список изученных слов"),
                BotCommand("help", "Справка по командам и кнопкам"),
                BotCommand("stop", "Отключить уведомления"),
                BotCommand("notifications_on", "Включить уведомления (5/день)"),
                BotCommand("reset", "Сбросить прогресс и начать заново"),
                BotCommand("add_word", "Добавить слово вручную"),
            ])
        except Exception as e:
            print(f"Could not set bot commands: {e}")
    
    try:
        polling_tasks = [
            asyncio.create_task(
                app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            ) for app in apps
        ]
        await stop_event.wait()

    finally:

        for app in apps:
            if app.updater.running:
                await app.updater.stop()

        for task in polling_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        for app in apps:
            await app.stop()
            await app.shutdown()


if __name__ == '__main__':

    running_exercises_file = 'running_exercise.jb' 
    running_activities = RunningActivities(running_exercises_file)

    user_data_root = os.getenv('CH_USER_DATA_ROOT')
    user_data_root = 'resources' if user_data_root is None else user_data_root
    user_data_root = Path(user_data_root)
    words_db_path = user_data_root / 'words_db.csv'
    decks_db_path = user_data_root / 'decks_db.csv'
    deck_word_db_path = user_data_root / 'deck_word.csv'
    words_progress_db_path = user_data_root / 'words_progress_db.csv'
    user_config_path = user_data_root / 'user_config.json'

    TIMEZONE = os.getenv('TIMEZONE')

    words_db = WordsDB(words_db_path)
    decks_db = DecksDB(decks_db_path, deck_word_db_path)
    words_progress_db = WordsProgressDB(words_progress_db_path)
    user_config = UserConfig(user_config_path)

    with open('resources/interface.json', 'r', encoding='utf-8') as fp:
        interface = json.loads(fp.read())

    templates = Templates(str(Path('resources/templates')))

    lp = LearningPlan(interface, templates, words_progress_db=words_progress_db, words_db=words_db, decks_db=decks_db, user_config=user_config)

    shared_objs = [user_config, words_db, words_progress_db, decks_db, running_activities]

    # list of known exercise buttons
    exercise_buttons = ['Hint', 'Correct answer', 'Answer audio', 'Next', 'Pronounce', 'Easier', 'Harder']

    apps = []
    lang_map = {}

    for bidx, (token, lang) in enumerate(zip(BOT_TOKENS, BOT_LANGS)):
        application = Application.builder().token(token).build()

        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_request))
        application.add_handler(CommandHandler("start", handle_command))
        application.add_handler(CommandHandler("help", handle_command))
        application.add_handler(CommandHandler("add_word", handle_command))
        application.add_handler(CommandHandler("next_test", handle_command))
        application.add_handler(CommandHandler("next_new", handle_command))
        application.add_handler(CommandHandler("stop", handle_command))
        application.add_handler(CommandHandler("notifications_off", handle_command))
        application.add_handler(CommandHandler("unsubscribe", handle_command))
        application.add_handler(CommandHandler("notifications_on", handle_command))
        application.add_handler(CommandHandler("subscribe", handle_command))
        application.add_handler(CommandHandler("reset", handle_command))
        application.add_handler(CommandHandler("reset_progress", handle_command))
        application.add_handler(CommandHandler("stats", handle_command))
        application.add_handler(CommandHandler("words", handle_command))
        application.add_handler(MessageHandler(filters.Regex(r"^/replay"), handle_command))
        application.add_handler(CallbackQueryHandler(handle_inline_request))

        job_queue = application.job_queue
        ping_interval = 15 * 60
        first_ping = nearest_start_time(ping_interval)
        print(f"First ping scheduled in: {first_ping} sec")
        
        job_queue.run_repeating(ping_users, interval=ping_interval, first=first_ping, job_kwargs={'misfire_grace_time': None})
        apps.append(application)
        lang_map[token] = lang

    asyncio.run(run_apps(apps))
