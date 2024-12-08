
import asyncio
import logging
import traceback
import os
import re
import time
import signal
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI
import aiosqlite
from private import API_ID, API_HASH, OPENAI_API_KEY, TG # Убедитесь, что этот файл существует и содержит ваши ключи

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

database_lock = asyncio.Lock()

#Обработчик сигналов для корректного завершения работы
def signal_handler(sig, frame):
    print('SYS_INFO: Выключение бота...')
    asyncio.run(shutdown())

async def shutdown():
    print("Закрытие ресурсов...")
    await app.stop()
    print("Выключение завершено.")
    exit(0)

# Функция создания и инициализации базы данных
async def create_database(db_file):
    db_file_path = os.path.abspath(db_file)
    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
    try:
        async with aiosqlite.connect(db_file_path) as db:
            await db.execute('''CREATE TABLE IF NOT EXISTS admins (user_id INTEGER PRIMARY KEY, username TEXT, level INTEGER DEFAULT 1)''')
            await db.execute('''CREATE TABLE IF NOT EXISTS banned_users (user_id INTEGER PRIMARY KEY, ban_until INTEGER, ban_reason TEXT, banned_by INTEGER)''')
            await db.execute('''CREATE TABLE IF NOT EXISTS ban_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, banned_by INTEGER, ban_until INTEGER, ban_reason TEXT, timestamp INTEGER)''')
            await db.execute('''CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY, username TEXT DEFAULT '-', first_name TEXT DEFAULT '-', last_name TEXT DEFAULT '-')''')
            await db.execute('''CREATE TABLE IF NOT EXISTS user_profile (user_id INTEGER PRIMARY KEY, model TEXT DEFAULT "gpt-4o")''') if db_file_path == os.path.abspath('bot_data.db') else await db.execute('''CREATE TABLE IF NOT EXISTS messages (user_id INTEGER, role TEXT, message TEXT)''')
            await db.commit()
        return True
    except Exception as e:
        logging.exception(f"Ошибка при создании базы данных {db_file}: {e}")
        return False


# Функции выполнения запросов к базе данных (без изменений)
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
    result = await execute_query(db_file, query, params)
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

# Функции работы с информацией о пользователях (без изменений)
async def update_user_info(user_id, user):
    username = user.username or '-'
    first_name = user.first_name or '-'
    last_name = user.last_name or '-'
    await execute_update('bot_data.db', "INSERT OR REPLACE INTO users (user_id, username, first_name, last_name) VALUES (?, ?, ?, ?)", (user_id, username, first_name, last_name))

async def is_admin(user_id, db_name='bot_data.db'):
    result = await execute_query_single(db_name, "SELECT level FROM admins WHERE user_id = ?", (user_id,))
    return result and result[0] >= 1

async def get_admin_level(user_id, db_name='bot_data.db'):
    result = await execute_query_single(db_name, "SELECT level FROM admins WHERE user_id = ?", (user_id,))
    return result[0] if result else 0

async def is_banned(user_id, db_name='bot_data.db'):
    result = await execute_query_single(db_name, "SELECT ban_until FROM banned_users WHERE user_id = ?", (user_id,))
    if result:
        ban_until = result[0]
        if ban_until == -1:
            return True
        if time.time() < ban_until:
            return True
        await execute_update(db_name, "DELETE FROM banned_users WHERE user_id = ?", (user_id,))
    return False

# Функции парсинга и форматирования времени (без изменений)
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

# Функция получения информации о пользователе из базы данных
async def get_user(user_id):
    result = await execute_query_single('bot_data.db', "SELECT username, first_name, last_name FROM users WHERE user_id = ?", (user_id,))
    if result:
        return {'username': result[0], 'first_name': result[1], 'last_name': result[2]}
    else:
        return None

# Функция получения ID пользователя по username или ID с учетом регистра
async def get_user_id(user_input, client):
    try:
        if user_input.isdigit():
            return int(user_input)
        else:
            # Приводим username к нижнему регистру перед поиском
            user_input_lower = user_input.lower()
            user = await client.get_users(user_input_lower)  # Поиск по нижнему регистру
            if user:
                await update_user_info(user.id, user)
                return user.id
            else:
                return None
    except Exception as e:
        logging.exception(f"Ошибка при получении user_id: {e}")
        return None

# Функции работы с моделью пользователя (без изменений)
async def set_user_model(user_id, model):
    await execute_update('_history.db', "INSERT OR IGNORE INTO user_profile (user_id, model) VALUES (?, ?)", (user_id, model))
    await execute_update('_history.db', "UPDATE user_profile SET model = ? WHERE user_id = ?", (model, user_id))

async def get_user_model(user_id):
    result = await execute_query_single('_history.db', "SELECT model FROM user_profile WHERE user_id = ?", (user_id,))
    return result[0] if result else "gpt-4o-mini"

# Функции работы с контекстом (без изменений)
async def get_context(user_id):
    rows = await execute_query('_history.db', "SELECT role, message FROM messages WHERE user_id = ? ORDER BY rowid ASC", (user_id,))
    return [{"role": row[0], "content": row[1]} for row in rows]

async def save_message(user_id, role, message):
    await execute_update('_history.db', "INSERT INTO messages (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    await execute_update('_history.db', """DELETE FROM messages WHERE user_id = ? AND rowid NOT IN (SELECT rowid FROM messages WHERE user_id = ? ORDER BY rowid DESC LIMIT 50)""", (user_id, user_id))

async def clear_context(user_id):
    await execute_update('_history.db', "DELETE FROM messages WHERE user_id = ?", (user_id,))

# Функция генерации ответа от OpenAI (без изменений)
async def generate_response(user_id, user_message):
    system_prompt = "Ты %name%-GPT, очень многофункциональный и крутой бот. Пока что у тебя есть gpt-4o и gpt-4o-mini, но в будущем будут новые фунцкции и фичи, тебя создал %name% (@%username%), если кто-то нашел баг, пусть пишут ему. Если ты не помнишь историю диалога то либо ее очистили с помощью команды /clear, либо это новый пользователь. Форматируй ответы в Markdown."
    try:
        context = await get_context(user_id)
        context.insert(0, {"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(model=await get_user_model(user_id), messages=context, temperature=0.5)
        response_message = response.choices[0].message.content
        await app.send_message(user_id, response_message, parse_mode="HTML")
        await save_message(user_id, "user", user_message)
        await save_message(user_id, "assistant", response_message)
        return response_message
    except Exception as e:
        logging.exception(f"Ошибка при генерации ответа: {e}")
        await app.send_message(user_id, "Произошла ошибка при генерации ответа. Пожалуйста, попробуйте позже.")
        return f"Произошла ошибка {e}"

#Класс для проверки типа документа (без изменений)
class IsDocument:
    def __call__(self, message):
        return message.document

#Функция загрузки и декодирования файлов (без изменений)
async def download_and_decode_file(file_id, file_path):
    try:
        file = await app.download_media(file_id) # Используем app для загрузки
        with open(file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        os.remove(file) # удаляем временный файл
        return file_content
    except UnicodeDecodeError:
        logging.exception("Ошибка декодирования файла.")
        return None
    except Exception as e:
        logging.exception(f"Ошибка загрузки/декодирования файла: {e}")
        return None

#Функция проверки авторизации
async def check_authorization(client):
    try:
        await client.get_me()
        print("Pyrogram client is authorized.")
        return True
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


async def main():
    global app, client
    app = Client("my_pyrogram_session", api_id=API_ID, api_hash=API_HASH)
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="http://localhost:1337/v1")
    try:
        if not await create_database('bot_data.db') or not await create_database('_history.db'):
            logging.critical("Критическая ошибка при создании баз данных. Завершение работы.")
            return

        await app.start()
        if not await check_authorization(app):
            logging.critical("Авторизация клиента Pyrogram не удалась. Завершение работы.")
            return

        # Обработчики сообщений
        @app.on_message(filters.command("start"))
        async def start_command(client, message):
            user_id = message.from_user.id
            await update_user_info(user_id, message.from_user)
            keyboard = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton("gpt-4o")], [KeyboardButton("gpt-4o-mini")]],
                resize_keyboard=True,
            )
            await app.send_message(
                message.chat.id,
                "Привет! Я твой AI помощник. Выбери модель или начни диалог.",
                reply_markup=keyboard,
            )

        @app.on_message(filters.text & filters.regex(r"^gpt-4o(-mini)?$"))
        async def set_model(client, message):
            try:
                await set_user_model(message.from_user.id, message.text)
                await app.send_message(message.chat.id, f"Модель успешно изменена на {message.text}.")
            except Exception as e:
                await app.send_message(message.chat.id, f"Ошибка изменения модели: {e}")

        @app.on_message(filters.command('ban_info'))
        async def ban_info(client, message):
            try:
                result = await execute_query_single('bot_data.db', "SELECT ban_until, ban_reason FROM banned_users WHERE user_id = ?", (message.from_user.id,))
                if result:
                    ban_until, ban_reason = result
                    await app.send_message(message.chat.id, f"{'Вы забанены навсегда.' if ban_until == -1 else f'Вы забанены до {datetime.fromtimestamp(ban_until)} по причине: {ban_reason}. Осталось времени: {format_time_left(ban_until)}.'}")
                else:
                    await app.send_message(message.chat.id, "Вы не забанены.")
            except Exception as e:
                logging.exception(f"Ошибка в ban_info: {e}")
                await app.send_message(message.chat.id, f"Произошла ошибка: {e}")

        @app.on_message(filters.command('ban'))
        async def ban_user(client, message):
            if not await is_admin(message.from_user.id):
                return await app.send_message(message.chat.id, "У вас нет прав для выполнения этой команды.")
            args = message.text.split(maxsplit=2)
            if len(args) < 2:
                return await app.send_message(message.chat.id, "Неверный формат команды. Используйте: /ban <@username|user_id> <reason>")

            # сохраняем исходный регистр для сообщения
            original_user_input = args[1]

            # Приводим username к нижнему регистру перед поиском
            user_input = args[1].lower()
            user_id = await get_user_id(user_input, app)
            if user_id is None:
                return await app.send_message(message.chat.id, f"Ошибка: Пользователь '{original_user_input}' не найден.")

            ban_reason = args[2] if len(args) > 2 else "Без указания причины"
            ban_until = -1

            async with database_lock:
                try:
                    await execute_update('bot_data.db', "INSERT OR IGNORE INTO banned_users (user_id, ban_until, ban_reason, banned_by) VALUES (?, ?, ?, ?)", (user_id, ban_until, ban_reason, message.from_user.id))
                    await execute_update('bot_data.db', "INSERT INTO ban_history (user_id, banned_by, ban_until, ban_reason, timestamp) VALUES (?, ?, ?, ?, ?)", (user_id, message.from_user.id, ban_until, ban_reason, int(time.time())))
                except Exception as e:
                    logging.exception(f"Ошибка базы данных при бане: {e}")
                    return await app.send_message(message.chat.id, f"Произошла ошибка при работе с базой данных: {e}")
            try:
                user_info = await get_user(user_id)
                username = user_info.get('username', f"ID: {user_id}")
                ban_message = f"Вы забанены админом {hbold('@' + message.from_user.username)} по причине: {ban_reason}"
                await app.send_message(user_id, ban_message, parse_mode="HTML")
                await app.send_message(message.chat.id, f"Пользователь {hbold(username)} забанен.", parse_mode="HTML")
            except Exception as e:
                logging.exception(f"Ошибка при уведомлении о бане: {e}")
                await app.send_message(message.chat.id, f"Произошла ошибка при уведомлении о бане: {e}")


        @app.on_message(filters.command('unban'))
        async def unban_user(client, message):
            if not await is_admin(message.from_user.id):
                return await app.send_message(message.chat.id, "У вас нет прав.")
            args = message.text.split()
            if len(args) < 2:
                return await app.send_message(message.chat.id, "Не указан пользователь для разбана. Используйте: /unban <@username|user_id>")
            user_input = args[1].lower()
            user_id = await get_user_id(user_input, app)
            if user_id is None:
                return await app.send_message(message.chat.id, f"Пользователь '{args[1]}' не найден.")
            async with database_lock:
                try:
                    success = await execute_update('bot_data.db', "DELETE FROM banned_users WHERE user_id = ?", (user_id,))
                    await app.send_message(message.chat.id, f"{'Пользователь с ID {user_id} разбанен.' if success else f'Ошибка удаления бана для пользователя с ID {user_id}.'}")
                except Exception as e:
                    logging.exception(f"Произошла ошибка в unban_user: {e}")
                    await app.send_message(message.chat.id, f"Произошла ошибка: {e}")

        @app.on_message(filters.command('admin'))
        async def admin_command(client, message):
            if not await is_admin(message.from_user.id) or not (await get_admin_level(message.from_user.id) >= 3):
                return await app.send_message(message.chat.id, "У вас нет прав.")
            args = message.text.split()
            if len(args) < 3:
                return await app.send_message(message.chat.id, "Недостаточно аргументов. Используйте: /admin add|remove <@username|user_id> [level]")
            action = args[1].lower()
            if action not in ["add", "remove"]:
                return await app.send_message(message.chat.id, "Неверная операция.")
            user_input = args[2]
            user_id = await get_user_id(user_input, app)
            if user_id is None:
                return await app.send_message(message.chat.id, f"Пользователь '{user_input}' не найден.")
            async with database_lock:
                try:
                    if action == "add":
                        level = int(args[3]) if len(args) > 3 and 1 <= int(args[3]) <= 3 else 1
                        try:
                            user_data = await app.get_chat(user_input) if user_input.startswith('@') else None
                            username = user_data.username if user_data else user_input
                            await execute_update('bot_data.db', "INSERT OR IGNORE INTO admins (user_id, username, level) VALUES (?, ?, ?)", (user_id, username, level))
                            await app.send_message(message.chat.id, f"Пользователь {username} добавлен в список администраторов с уровнем {level}.")
                        except Exception as e:
                            await app.send_message(message.chat.id, f"Ошибка получения данных пользователя: {e}")
                    elif action == "remove":
                        success = await execute_update('bot_data.db', "DELETE FROM admins WHERE user_id = ?", (user_id,))
                        await app.send_message(message.chat.id, f"{'Пользователь {user_input} удалён из списка администраторов.' if success else 'Ошибка удаления пользователя из базы данных.'}")
                except Exception as e:
                    logging.exception(f"Произошла ошибка в команде /admin: {e}")
                    await app.send_message(message.chat.id, f"Произошла ошибка: {e}")

        @app.on_message(filters.command('setadminlevel'))
        async def set_admin_level(client, message):
            if not await is_admin(message.from_user.id) or not (await get_admin_level(message.from_user.id) >= 3):
                return await app.send_message(message.chat.id, "У вас нет прав.")
            args = message.text.split()
            if len(args) < 3:
                return await app.send_message(message.chat.id, "Недостаточно аргументов.  Используйте: /setadminlevel <user_id> <level>")
            try:
                user_id = int(args[1])
                level = int(args[2])
                if not 1 <= level <= 3:
                    return await app.send_message(message.chat.id, "Уровень доступа должен быть от 1 до 3.")
                async with database_lock:
                    success = await execute_update('bot_data.db', "UPDATE admins SET level = ? WHERE user_id = ?", (level, user_id))
                    await app.send_message(message.chat.id, f"{'Уровень доступа для пользователя {user_id} изменён на {level}.' if success else f'Ошибка обновления уровня доступа для пользователя {user_id}.'}")
            except (ValueError, IndexError) as e:
                await app.send_message(message.chat.id, f"Ошибка: {e}")
            except Exception as e:
                logging.exception(f"Ошибка в команде /setadminlevel: {e}")
                await app.send_message(message.chat.id, f"Произошла неизвестная ошибка: {e}")

        @app.on_message(filters.command('admins'))
        async def list_admins(client, message):
            if not await is_admin(message.from_user.id):
                return await app.send_message(message.chat.id, "У вас нет прав.")
            try:
                admins = await execute_query('bot_data.db', "SELECT user_id, username, level FROM admins")
                await app.send_message(message.chat.id, f"{'Список администраторов:\n' + '\n'.join([f'ID: {admin[0]}, Username: @{admin[1]}, Уровень доступа: {admin[2]}' for admin in admins]) if admins else 'Список администраторов пуст.'}")
            except Exception as e:
                logging.exception(f"Ошибка в команде /admins: {e}")
                await app.send_message(message.chat.id, f"Произошла ошибка: {e}")

        @app.on_message(filters.command('banhistory'))
        async def ban_history(client, message):
            if not await is_admin(message.from_user.id):
                return await app.send_message(message.chat.id, "У вас нет прав.")
            try:
                history = await execute_query('bot_data.db', "SELECT * FROM ban_history")
                history_str = ""
                for item in history:
                    try:
                        user = await app.get_chat(item[1])
                        username = user.username if user else "Unknown"
                    except Exception as e:
                        username = "Unknown (user deleted or blocked)"
                    try:
                        banned_by = await app.get_chat(item[2])
                        banned_by_username = banned_by.username if banned_by else "Unknown"
                    except Exception as e:
                        banned_by_username = "Unknown (user deleted or blocked)"
                    time_str = datetime.fromtimestamp(item[5]).strftime('%Y-%m-%d %H:%M:%S')
                    history_str += f"ID: {item[0]}, User: {username}, Banned by: {banned_by_username}, Time: {time_str}, Reason: {item[4]}\n"
                await app.send_message(message.chat.id, f"{'История банов:\n' + history_str if history_str else 'История банов пуста.'}")
            except Exception as e:
                logging.exception(f"Ошибка в команде /banhistory: {e}")
                await app.send_message(message.chat.id, f"Произошла ошибка: {e}")

        @app.on_message(filters.command("clear"))
        async def clear_command(client, message):
            args = message.text.split()
            if len(args) > 1:
                if not await is_admin(message.from_user.id) or not (await get_admin_level(message.from_user.id) >= 2):
                    return await app.send_message(message.chat.id, "У вас нет прав.")
                user_input = args[1]
                if not (user_input.isdigit() or re.match(r'^@\w+$', user_input)):
                    return await app.send_message(message.chat.id, "Неверный формат user_id или username.")
                user_id = await get_user_id(user_input, app)
                if user_id is None:
                    return await app.send_message(message.chat.id, "Пользователь не найден.")
                admin_name = message.from_user.username
            else:
                user_id = message.from_user.id
                admin_name = None
            async with database_lock:
                try:
                    success = await execute_update('_history.db', "DELETE FROM messages WHERE user_id = ?", (user_id,))
                    if success:
                        await app.send_message(message.chat.id, f"{'Контекст пользователя @{user_input} очищен админом @{admin_name}.' if admin_name else 'Ваш контекст успешно очищен.'}")
                        if admin_name: await app.send_message(user_id, f"Ваш контекст был очищен админом @{admin_name}.")
                    else:
                        await app.send_message(message.chat.id, "Ошибка очистки контекста.")
                except Exception as e:
                    logging.exception(f"Произошла ошибка при очистке контекста: {e}")
                    await app.send_message(message.chat.id, "Произошла ошибка при очистке контекста. Попробуйте позже.")


        @app.on_message(IsDocument())
        async def handle_document(client, message):
            mime_type = message.document.mime_type
            file_id = message.document.file_id
            try:
                file = await app.download_media(file_id)
                if not mime_type.startswith("text/"):
                    return await app.send_message(message.chat.id, "Пожалуйста, отправьте текстовый файл.")
                file_content = await download_and_decode_file(file, file_id)
                if not file_content:
                    return await app.send_message(message.chat.id, "Не могу обработать файл. Проверьте кодировку.")
                truncated_content = file_content[:50000000]
                response_text = await generate_response(message.from_user.id, truncated_content)
                await app.send_message(message.chat.id, response_text, parse_mode="HTML")
            except Exception as e:
                await app.send_message(message.chat.id, f"Произошла ошибка: {e}")
                logging.exception(e)

        @app.on_message()
        async def handle_all_messages(client, message):
            if await is_banned(message.from_user.id):
                return await app.send_message(message.chat.id, "Вы забанены!")
            if message.text:
                await generate_response(message.from_user.id, message.text)

    except Exception as e:
        logging.exception(f"Критическая ошибка в main(): {e}")
    finally:
        await app.stop()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())

