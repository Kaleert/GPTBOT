#
import asyncio
import logging
import traceback
import os
import re
import time
import signal
import aiohttp
from datetime import datetime, timedelta
from pyrogram import Client, errors
import markdown2
from aiogram import Bot, Dispatcher, types
from aiogram.enums.chat_member_status import ChatMemberStatus
from aiogram.enums.parse_mode import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, BaseFilter
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from openai import OpenAI, OpenAIError
import aiosqlite
from private import TG, OPENAI_API_KEY, API_ID, API_HASH, system_prompt
import sqlite3


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
pyrogram_client = Client("KaleertGPT", api_id=API_ID, api_hash=API_HASH)
database_lock = asyncio.Lock()

def signal_handler(sig, frame):
    print('SYS_INFO: Выключение бота...')
    asyncio.run(shutdown())  # Вызываем функцию shutdown()

async def shutdown():
    print("Закрытие ресурсов...")
    if pyrogram_client:
        asyncio.create_task(pyrogram_client.stop()) # Запускаем асинхронно

    await bot.session.close()
    print("Выключение завершено.")
    exit(0)


async def check_authorization(client):
    try:
        await client.get_me()
        print("Pyrogram client is authorized.")
        return True
    except errors.AuthKeyInvalid as e:
        print(f"Pyrogram client is NOT authorized: {e}")
        return False
    except errors.RPCError as e:
        print(f"Pyrogram RPC error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

async def create_database(db_file):
    db_file_path = os.path.abspath(db_file)
    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
    try:
        async with aiosqlite.connect(db_file_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS admins (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    level INTEGER DEFAULT 1
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS banned_users (
                    user_id INTEGER PRIMARY KEY,
                    ban_until INTEGER,
                    ban_reason TEXT,
                    banned_by INTEGER
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS ban_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    banned_by INTEGER,
                    ban_until INTEGER,
                    ban_reason TEXT,
                    timestamp INTEGER
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT DEFAULT '-',
                    first_name TEXT DEFAULT '-',
                    last_name TEXT DEFAULT '-'
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id INTEGER PRIMARY KEY,
                    model TEXT DEFAULT "gpt-4o"
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    user_id INTEGER,
                    role TEXT,
                    message TEXT
                )
            ''')
            await db.commit()
        return True
    except Exception as e:
        logging.exception(f"Ошибка при создании базы данных {db_file}: {e}")
        return False

async def execute_query(db_file, query, params=()):
    try:
        async with aiosqlite.connect(db_file) as db:
            async with db.cursor() as cursor:
                await cursor.execute(query, params)
                return await cursor.fetchall()
    except Exception as e:
        logging.exception(f"Ошибка SQL: {e}")
        return None

async def execute_query_single(db_file, query, params=()):
    logging.debug(f"Executing query: {query} with params: {params}")
    result = await execute_query(db_file, query, params)
    logging.debug(f"Query result: {result}")
    return result[0] if result else None

async def execute_update(db_file, query, params=()):
    try:
        async with aiosqlite.connect(db_file) as db:
            async with db.cursor() as cursor:
                await cursor.execute(query, params)
                await db.commit()
                return True
    except Exception as e:
        logging.exception(f"Ошибка SQL: {e}")
        return False
        
async def update_user_info(user_id, user: types.User):
    username = user.username or '-'
    first_name = user.first_name or '-'
    last_name = user.last_name or '-'
    await execute_update('bot_data.db', "INSERT OR REPLACE INTO users (user_id, username, first_name, last_name) VALUES (?, ?, ?, ?)", (user_id, username.lower() if username else '-', first_name, last_name)) # username.lower()

async def is_admin(user_id, db_name='bot_data.db'):
    try:
        result = await execute_query_single(db_name, "SELECT level FROM admins WHERE user_id = ?", (user_id,))
        return result and result[0] >= 1
    except Exception as e:
        logging.exception(f"Ошибка в is_admin: {e}")
        return False

async def get_admin_level(user_id, db_name='bot_data.db'):
    try:
        result = await execute_query_single(db_name, "SELECT level FROM admins WHERE user_id = ?", (user_id,))
        return result[0] if result else 0
    except Exception as e:
        logging.exception(f"Ошибка в get_admin_level: {e}")
        return 0

async def is_banned(user_id, db_name='bot_data.db'):
    try:
        async with aiosqlite.connect(db_name) as db:
            async with db.cursor() as cursor:
                await cursor.execute("SELECT ban_until FROM banned_users WHERE user_id = ?", (user_id,))
                result = await cursor.fetchone()
                if result:
                    ban_until = result[0]
                    if ban_until == -1:
                        return True
                    if time.time() < ban_until:
                        return True
                    await cursor.execute("DELETE FROM banned_users WHERE user_id = ?", (user_id,))
                    await db.commit()
                return False
    except Exception as e:
        logging.exception(f"Ошибка в is_banned: {e}")
        return False

def parse_duration(duration_text):
    time_units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'y': 31536000}
    match = re.match(r'^(\d+)([smhdwy]?)$', duration_text)
    if match:
        value, unit = match.groups()
        try:
            value = int(value)
            unit = unit or 's'
            return value * time_units[unit] if unit in time_units else None
        except ValueError:
            logging.error(f"Некорректное числовое значение в duration: {duration_text}")
            return None
    elif duration_text.lower() == 'forever':
        return -1
    else:
        logging.error(f"Неверный формат duration: {duration_text}")
        return None

def format_time_left(ban_until):
    if ban_until == -1:
        return "навсегда"
    time_left = ban_until - time.time()
    if time_left <= 0:
        return "бан уже закончился"
    seconds = int(time_left)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    result = []
    if days: result.append(f"{days} дн.")
    if hours: result.append(f"{hours} ч.")
    if minutes: result.append(f"{minutes} мин.")
    if seconds and not (days or hours or minutes): result.append(f"{seconds} сек.")
    return ", ".join(result) or "менее 1 секунды"

def hbold(text):
    return f"<b>{text}</b>"

async def get_user(user_id):
    result = await execute_query_single('bot_data.db', "SELECT username, first_name, last_name FROM users WHERE user_id = ?", (user_id,))
    if result:
        return {'username': result[0], 'first_name': result[1], 'last_name': result[2]}
    else:
        return None
        
async def get_user_id(user_input, bot: Bot):
    logging.debug(f"get_user_id called with user_input: {user_input}")
    try:
        user_input_lower = user_input.lower() # Преобразуем к нижнему регистру для поиска в базе данных

        if user_input_lower.isdigit():
            user_id = int(user_input_lower)
            logging.debug(f"User ID received directly: {user_id}")
            return user_id
        elif user_input_lower.startswith('@'):
            username = user_input_lower[1:]  # Убираем @
            logging.debug(f"Searching for user by username: {username} in database")
            result = await execute_query_single('bot_data.db', "SELECT user_id FROM users WHERE username = ?", (username,))
            if result:
                user_id = result[0]
                logging.debug(f"User ID found in database: {user_id}")
                return user_id
            else:
                logging.debug(f"User not found in database, querying Pyrogram for username: {username}")
                async with pyrogram_client:
                    try:
                        user = await pyrogram_client.get_users(username) # Pyrogram сам разберется с регистром
                        if user:
                            user_id = user.id
                            logging.debug(f"User ID received from Pyrogram: {user_id}")
                            await update_user_info(user_id, user)
                            return user_id
                        else:
                            logging.warning(f"User '{username}' not found using Pyrogram.")
                            return None
                    except errors.UsernameNotOccupied:
                        logging.warning(f"Username '{username}' is not occupied.")
                        return None
                    except errors.UsernameInvalid:
                        logging.warning(f"Invalid username '{username}'.")
                        return None
                    except errors.RPCError as e:
                        logging.error(f"Pyrogram RPC error: {e}")
                        return None
                    except Exception as e:
                        logging.exception(f"An unexpected error occurred during Pyrogram query: {e}")
                        return None
        else:
            logging.debug(f"Invalid user input format: {user_input}")
            return None

    except Exception as e:
        logging.exception(f"Error in get_user_id: {e}")
        return None
        
async def set_user_model(user_id, model):
    await execute_update('_history.db', "INSERT OR IGNORE INTO user_profile (user_id, model) VALUES (?, ?)", (user_id, model))
    await execute_update('_history.db', "UPDATE user_profile SET model = ? WHERE user_id = ?", (model, user_id))

async def get_user_model(user_id):
    result = await execute_query_single('_history.db', "SELECT model FROM user_profile WHERE user_id = ?", (user_id,))
    return result[0] if result else "gpt-4o-mini"

async def get_context(user_id):
    rows = await execute_query('_history.db', "SELECT role, message FROM messages WHERE user_id = ? ORDER BY rowid ASC", (user_id,))
    return [{"role": row[0], "content": row[1]} for row in rows]

async def save_message(user_id, role, message):
    await execute_update('_history.db', "INSERT INTO messages (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    await execute_update('_history.db', """DELETE FROM messages WHERE user_id = ? AND rowid NOT IN (SELECT rowid FROM messages WHERE user_id = ? ORDER BY rowid DESC LIMIT 50)""", (user_id, user_id))

async def clear_context(user_id):
    await execute_update('_history.db', "DELETE FROM messages WHERE user_id = ?", (user_id,))
    
def escape_markdown(text):
    """Экранирует символы для Markdown с использованием markdown2."""
    try:
        return markdown2.markdown(text)
    except Exception as e:
        logging.error(f"Ошибка при обработке Markdown: {e}")
        return text # Вернуть оригинальный текст, если произошла ошибка во время обработки Markdown.
    
def sanitize_markdown(text):
    # Удаляем или экранируем проблемные символы
    return text.replace('*', '\\*').replace('_', '\\_')

async def start_typing(bot: Bot, user_id: int):
    """Запускает индикатор 'печатает...' в Telegram."""
    try:
        await bot.send_chat_action(chat_id=user_id, action="typing")
    except TelegramAPIError as e:
        logging.warning(f"Ошибка при отправке индикатора 'печатает...': {e}")

async def stop_typing(bot: Bot, user_id: int):
    """Останавливает индикатор 'печатает...'."""
    try:
        await bot.send_chat_action(chat_id=user_id, action="cancel")
    except TelegramAPIError as e:
        logging.warning(f"Ошибка при остановке индикатора 'печатает...': {e}")

async def send_message_with_retry(bot, chat_id, text, parse_mode=ParseMode.MARKDOWN):
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            await bot.send_message(chat_id, text, parse_mode=parse_mode)
            return True
        except TelegramAPIError as e:
            logging.warning(f"Ошибка Telegram API (попытка {attempt}/{max_retries}): {e}. Повторяем...")
            await asyncio.sleep(attempt * 2)  # Экспоненциальный backoff
    logging.error(f"Не удалось отправить сообщение после нескольких попыток: {text}")
    return False

def split_message(message, max_length):
    """Разбивает длинное сообщение на части."""
    parts = []
    for i in range(0, len(message), max_length):
        parts.append(message[i:i + max_length])
    return parts

async def generate_response(user_id, user_message, bot: Bot, client: OpenAI):
    try:
        await start_typing(bot, user_id)
        _model = await get_user_model(user_id)
        context = await get_context(user_id)
        context.insert(0, {"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(model=_model, messages=context, temperature=0.5)
        response_message = response.choices[0].message.content

        # Санитация Markdown (удаление  тегов)
        response_message = re.sub(r'<p>(.*?)', r'\1', response_message, flags=re.DOTALL)
        response_message = response_message.strip()

        MAX_MESSAGE_LENGTH = 4096  # Максимальная длина сообщения Telegram
        response_parts = split_message(response_message, MAX_MESSAGE_LENGTH)

        for part in response_parts:
            await send_message_with_retry(bot, user_id, sanitize_markdown(part), parse_mode="Markdown")
        await stop_typing(bot, user_id)
        await save_message(user_id, "user", user_message)
        await save_message(user_id, "assistant", response_message)
        return response_message
    except OpenAIError as openai_error:
        await stop_typing(bot, user_id)
        logging.exception(f"Ошибка OpenAI API: {openai_error}")
        await send_message_with_retry(bot, user_id, "Произошла ошибка OpenAI API. Попробуйте позже.", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await stop_typing(bot, user_id)
        logging.exception(f"Ошибка при генерации ответа: {e}")
        await send_message_with_retry(bot, user_id, "Произошла неизвестная ошибка. Попробуйте позже.", parse_mode=ParseMode.MARKDOWN)


class IsDocument(BaseFilter):
    async def __call__(self, message: types.Message) -> bool:
        return message.content_type == types.ContentType.DOCUMENT
        
async def download_and_decode_file(file_id, file_path, bot: Bot):
    try:
        file = await bot.get_file(file_id)
        bytes_io = await bot.download_file(file.file_path)
        return bytes_io.read().decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        logging.exception("Ошибка декодирования файла.")
        return None
    except TelegramAPIError as e:
        logging.exception(f"Ошибка Telegram API: {e}")
        return None
    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")
        return None
        
class IsDocument(BaseFilter):
    async def __call__(self, message: types.Message) -> bool:
        return message.content_type == types.ContentType.DOCUMENT
        
async def download_and_decode_file(file_id, file_path, bot: Bot):
    try:
        file = await bot.get_file(file_id)
        bytes_io = await bot.download_file(file.file_path)
        return bytes_io.read().decode('utf-8', errors='ignore')
    except UnicodeDecodeError:
        logging.exception("Ошибка декодирования файла.")
        return None
    except TelegramAPIError as e:
        logging.exception(f"Ошибка Telegram API: {e}")
        return None
    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")
        return None
        
async def main():
    global pyrogram_client, bot, dp, client
    pyrogram_client = Client("Kaleert", api_id=API_ID, api_hash=API_HASH)
    try:
        if not await create_database('bot_data.db') or not await create_database('_history.db'):
            logging.critical("Критическая ошибка при создании баз данных. Завершение работы.")
            return

        bot = Bot(token=TG)
        dp = Dispatcher()
        dp.bot = bot
        client = OpenAI(api_key=OPENAI_API_KEY, base_url="http://localhost:1337/v1")
        keyboard = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="gpt-4o")], [KeyboardButton(text="gpt-4o-mini")]], resize_keyboard=True)
        await pyrogram_client.start()
        if not await check_authorization(pyrogram_client):
            logging.critical("Авторизация клиента Pyrogram не удалась. Завершение работы.")
            return

        #Handlers
        @dp.message(Command("start"))
        async def start_command(message: types.Message):
            await update_user_info(message.from_user.id, message.from_user)
            await message.answer("Привет! Я твой AI помощник. Выбери модель или начни диалог.", reply_markup=keyboard)

        @dp.message(lambda message: message.text in ["gpt-4o", "gpt-4o-mini"])
        async def set_model(message: types.Message):
            try:
                await set_user_model(message.from_user.id, message.text)
                await message.answer(f"Модель успешно изменена на {message.text}.")
            except Exception as e:
                await message.reply(f"Ошибка изменения модели: {e}")

        @dp.message(Command('ban_info'))
        async def ban_info(message: types.Message):
            try:
                result = await execute_query_single('bot_data.db', "SELECT ban_until, ban_reason FROM banned_users WHERE user_id = ?", (message.from_user.id,))
                if result:
                    ban_until, ban_reason = result
                    await message.reply(f"{'Вы забанены навсегда.' if ban_until == -1 else f'Вы забанены до {datetime.fromtimestamp(ban_until)} по причине: {ban_reason}. Осталось времени: {format_time_left(ban_until)}.'}")
                else:
                    await message.reply("Вы не забанены.")
            except Exception as e:
                logging.exception(f"Ошибка в ban_info: {e}")
                await message.reply(f"Произошла ошибка: {e}")

        @dp.message(Command('ban'))
        async def ban_user(message: types.Message):
            if not await is_admin(message.from_user.id):
                return await message.reply("У вас нет прав.")
            args = message.text.split(maxsplit=3)
            if len(args) < 3:
                return await message.reply("Не указана причина бана или время бана. Используйте: /ban <@username|user_id> <duration (m, h, d, forever)> <reason>")
            try:
                user_id = await get_user_id(args[1], bot)
                user_name = args[1]
                if user_id is None:
                    return await message.reply(f"Пользователь 'user_name' не найден.")
                duration_text = args[2]
                ban_reason = " ".join(args[3:])
                duration = parse_duration(duration_text)
                if duration is None or duration == 0:
                    return await message.reply("Неверный формат времени бана.")
                now = datetime.now()
                ban_until = int((now + timedelta(seconds=duration) if duration != -1 else datetime.max).timestamp())
                async with database_lock:
                    try:
                        await execute_update('bot_data.db', "INSERT OR IGNORE INTO banned_users (user_id, ban_until, ban_reason, banned_by) VALUES (?, ?, ?, ?)", (user_id, ban_until, ban_reason, message.from_user.id))
                        await execute_update('bot_data.db', "INSERT INTO ban_history (user_id, banned_by, ban_until, ban_reason, timestamp) VALUES (?, ?, ?, ?, ?)", (user_id, message.from_user.id, ban_until, ban_reason, int(time.time())))
                    except Exception as e:
                        logging.exception(f"Произошла ошибка в ban_user (база данных): {e}")
                        return await message.reply(f"Произошла ошибка при работе с базой данных: {e}")
                try:
                    await bot.send_message(user_id, f"Вас забанил @{message.from_user.username} на {format_time_left(ban_until)} по причине:\n{ban_reason}")
                except TelegramAPIError as e:
                    logging.warning(f"Не удалось отправить сообщение пользователю {user_name}: {e}")
                    await message.reply(f"Пользователь с ID {user_name} забанен, но сообщение не отправлено.", parse_mode="HTML")
                await message.reply(f"Пользователь с ID {user_name} забанен на {format_time_left(ban_until)} по причине:\n{ban_reason}")
            except Exception as e:
                logging.exception(f"Произошла ошибка в ban_user (другая ошибка): {e}")
                await message.reply(f"Произошла ошибка: {e}")

        @dp.message(Command('unban'))
        async def unban_user(message: types.Message):
            if not await is_admin(message.from_user.id):
                return await message.reply("У вас нет прав.")
            args = message.text.split()
            if len(args) < 2:
                return await message.reply("Не указан пользователь для разбана. Используйте: /unban <@username|user_id>")
            try:
                user_id = await get_user_id(args[1], bot)
                user_name = args[1]
                if user_id is None:
                    return await message.reply(f"Пользователь '{user_name}' не найден.")
                async with database_lock:
                    try:
                        success = await execute_update('bot_data.db', "DELETE FROM banned_users WHERE user_id = ?", (user_id,))
                        if success:
                            await message.reply(f"Пользователь с ID '{user_name}' разбанен.")  
                        else:
                            await message.reply(f"Ошибка удаления бана для пользователя с ID {user_name}.")
                    except Exception as e:
                        logging.exception(f"Произошла ошибка в unban_user: {e}")
                        await message.reply(f"Произошла ошибка: {e}")
            except Exception as e:
                logging.exception(f"Ошибка при получении user_id в unban_user: {e}")
                await message.reply(f"Произошла ошибка: {e}")

        @dp.message(Command('admin'))
        async def admin_command(message: types.Message):
            if not await is_admin(message.from_user.id) or not (await get_admin_level(message.from_user.id) >= 3):
                return await message.reply("У вас нет прав.")

            args = message.text.split()
            if len(args) < 2:
                return await message.reply("Недостаточно аргументов. Используйте: /admin add|remove <@username|user_id>")

            action = args[1].lower()
    
            if action == "add":
                if len(args) < 3:
                    return await message.reply("Не указан пользователь для добавления. Используйте: /admin add <@username|user_id> [level]")
                user_input = args[2]
                level = int(args[3]) if len(args) > 3 and 1 <= int(args[3]) <= 3 else 1
                user_id = await get_user_id(user_input, bot)

                if user_id is None:
                    return await message.reply(f"Пользователь '{user_input}' не найден.")

                async with database_lock:
                    try:
                        username = user_input[1:] if user_input.startswith('@') else user_input  # Убираем @, если есть
                        await execute_update('bot_data.db', "INSERT OR IGNORE INTO admins (user_id, username, level) VALUES (?, ?, ?)", (user_id, username, level))
                        await message.reply(f"Пользователь @{username} добавлен в список администраторов с уровнем {level}.")
                    except Exception as e:
                        logging.exception(f"Ошибка в добавлении администратора: {e}")
                        await message.reply(f"Произошла ошибка при добавлении администратора: {e}")

            elif action == "remove":
                if len(args) < 3:
                    return await message.reply("Не указан пользователь для удаления. Используйте: /admin remove <@username|user_id>")
                user_input = args[2]
                user_id = await get_user_id(user_input, bot)

                if user_id is None:
                    return await message.reply(f"Пользователь '{user_input}' не найден.")

                async with database_lock:
                    try:
                        success = await execute_update('bot_data.db', "DELETE FROM admins WHERE user_id = ?", (user_id,))
                        await message.reply(f"{'Пользователь удалён из списка администраторов.' if success else 'Ошибка удаления пользователя из базы данных.'}")
                    except Exception as e:
                        logging.exception(f"Ошибка в удалении администратора: {e}")
                        await message.reply(f"Произошла ошибка при удалении администратора: {e}")

                    else:
                        return await message.reply("Неверная операция. Используйте: /admin add|remove <@username|user_id>")
                
        @dp.message(Command('admins'))
        async def list_admins(message: types.Message):
            if not await is_admin(message.from_user.id):
                return await message.reply("У вас нет прав.")
            try:
                admins = await execute_query('bot_data.db', "SELECT user_id, username, level FROM admins")
                await message.reply(f"{'Список администраторов:\n' + '\n'.join([f'ID: {admin[0]}, Username: @{admin[1]}, Уровень доступа: {admin[2]}' for admin in admins]) if admins else 'Список администраторов пуст.'}")
            except Exception as e:
                logging.exception(f"Ошибка в команде /admins: {e}")
                await message.reply(f"Произошла ошибка: {e}")
                
        @dp.message(Command('banhistory'))
        async def ban_history(message: types.Message):
            if not await is_admin(message.from_user.id):
                return await message.reply("У вас нет прав.")
            try:
                history = await execute_query('bot_data.db', "SELECT * FROM ban_history")
                history_str = ""
                for item in history:
                    try:
                        user = await bot.get_chat(item[1])
                        username = user.username if user else "Unknown"
                    except TelegramAPIError:
                        username = "Unknown (user deleted or blocked)"
                    try:
                        banned_by = await bot.get_chat(item[2])
                        banned_by_username = banned_by.username if banned_by else "Unknown"
                    except TelegramAPIError:
                        banned_by_username = "Unknown (user deleted or blocked)"
                    time_str = datetime.fromtimestamp(item[5]).strftime('%Y-%m-%d %H:%M:%S')
                    history_str += f"ID: {item[0]}, User: {username}, Banned by: {banned_by_username}, Time: {time_str}, Reason: {item[4]}\n"
                await message.reply(f"{'История банов:\n' + history_str if history_str else 'История банов пуста.'}")
            except Exception as e:
                logging.exception(f"Ошибка в команде /banhistory: {e}")
                await message.reply(f"Произошла ошибка: {e}")

        @dp.message(Command("clear"))
        async def clear_command(message: types.Message):
            args = message.text.split()
            try:
                if len(args) > 1:
                    if not await is_admin(message.from_user.id) or not (await get_admin_level(message.from_user.id) >= 2):
                        return await message.reply("У вас нет прав.")
                    user_input = args[1]
                    if not (user_input.isdigit() or re.match(r'^@\w+$', user_input)):
                        return await message.reply("Неверный формат user_id или username.")
                    user_id = await get_user_id(user_input, bot)
                    if user_id is None:
                        return await message.reply("Пользователь не найден.")
                    admin_name = message.from_user.username
                else:
                    user_id = message.from_user.id
                    admin_name = None

                async with database_lock:
                    try:
                        success = await execute_update('_history.db', "DELETE FROM messages WHERE user_id = ?", (user_id,))
                        if success:
                            await message.reply(f"{'Контекст пользователя ' + (f'{user_input} ' if admin_name else '') + 'очищен' + (f' админом @{admin_name}' if admin_name else '') + '.'}")
                            if admin_name:
                                try:
                                    await bot.send_message(user_id, f"Ваш контекст был очищен админом @{admin_name}.")
                                except TelegramAPIError as e:
                                    logging.error(f"Ошибка отправки сообщения пользователю {user_id}: {e}")
                        else:
                            await message.reply("Ошибка очистки контекста.")
                    except Exception as e:
                        logging.exception(f"Произошла ошибка при очистке контекста: {e}")
                        await message.reply("Произошла ошибка при очистке контекста. Попробуйте позже.")
            except Exception as e:
                logging.exception(f"Произошла общая ошибка в clear_command: {e}")
                await message.reply(f"Произошла ошибка: {e}")
                    
        @dp.message(IsDocument())
        async def handle_document(message: types.Message):
            if await is_banned(message.from_user.id):
                return await message.reply("Вы забанены!")
            mime_type = message.document.mime_type
            file_id = message.document.file_id
            file_size = message.document.file_size

            try:
                # Проверка размера файла
                max_file_size = 6000000  # 6 МБ
                if file_size > max_file_size:
                    return await message.reply(f"Размер файла слишком большой. Максимальный размер: {max_file_size / (1024 * 1024):.2f} MB", parse_mode="Markdown")

                file = await bot.get_file(file_id)
                file_path = file.file_path
                if not mime_type.startswith("text/"):
                    return await message.reply("Пожалуйста, отправьте текстовый файл.", parse_mode="Markdown")
                file_content = await download_and_decode_file(file_id, file_path, bot)
                if file_content is None:
                    return await message.reply("Не могу обработать файл. Проверьте кодировку и тип файла.", parse_mode="Markdown")
                truncated_content = file_content[:50000000]  # Ограничение размера текста остается
                response_text = await generate_response(message.from_user.id, truncated_content, bot, client)
            except TelegramAPIError as e:
                await message.reply(f"Ошибка Telegram API: {e}")
            except Exception as e:
                await message.reply(f"Произошла ошибка: {e}")
                logging.exception(e)
                
        @dp.message()
        async def handle_all_messages(message: types.Message):
            if await is_banned(message.from_user.id):
                return await message.reply("Вы забанены!")
            if isinstance(message, types.ChatMemberUpdated):
                if message.new_chat_member.user.id != bot.id and message.old_chat_member.status != message.new_chat_member.status:
                    await update_user_info(message.new_chat_member.user.id, message.new_chat_member.user)
            elif message.text:
                await generate_response(message.from_user.id, message.text, bot, client)

        workdir="./pyrogram" # Создайте эту папку вручную
        async with Client("my_pyrogram_session", api_id=API_ID, api_hash=API_HASH, workdir=workdir) as p:
            pyrogram_client = p
            try:
                if not await check_authorization(pyrogram_client):
                    logging.critical("Авторизация клиента Pyrogram не удалась. Завершение работы.")
                    return

                await bot.get_me()
                try:
                    await dp.start_polling(bot) # Вернулись к стандартному start_polling
                except Exception as e:
                    logging.exception(f"Ошибка во время опроса бота: {e}")
            except sqlite3.OperationalError as e:
                logging.error(f"Ошибка доступа к базе данных Pyrogram: {e}. Попробуйте остановить предыдущий процесс.")
                await shutdown()
            except Exception as e:
                logging.exception(f"Ошибка при работе с Pyrogram: {e}")
                await shutdown()

    except Exception as e:
        logging.critical(f"Необработанная ошибка в main(): {e}")
        traceback.print_exc()
        await shutdown()

    finally:
        await bot.session.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())
