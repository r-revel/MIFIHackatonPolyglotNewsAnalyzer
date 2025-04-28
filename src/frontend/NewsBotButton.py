from dotenv import load_dotenv

from telegram import Update, InputFile
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    ConversationHandler
)

document_filter = (
        filters.Document.FileExtension("txt") |
        filters.Document.FileExtension("csv"))

import requests
import logging
import os
import io

import nest_asyncio

nest_asyncio.apply()

load_dotenv()
API_ENDPOINT = os.getenv("API_URL") + "/predict"

# Настройка логгера
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния диалога
MAIN_MENU, WAITING_TEXT, WAITING_FILE = range(3)

# Тексты кнопок
BTN_TEXT_LENGTH = '📏 Категория новости'
BTN_FILE_LENGTH = '📁 Файл с новостями'
BTN_CANCEL = '❌ Отмена'


# Удаляет предыдущие информационные сообщения
async def delete_previous_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'messages_to_delete' in context.user_data:
        for msg_id in context.user_data['messages_to_delete']:
            try:
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=msg_id
                )
            except Exception as e:
                logger.error(f"Ошибка удаления сообщения: {e}")
        context.user_data['messages_to_delete'] = []


# Собирает информационные сообщения от бота
async def track_message(context: ContextTypes.DEFAULT_TYPE, message_id: int):
    if 'messages_to_delete' not in context.user_data:
        context.user_data['messages_to_delete'] = []
    context.user_data['messages_to_delete'].append(message_id)


# Сбор информационных сообщений от пользователя
async def track_user_message(context: ContextTypes.DEFAULT_TYPE, message_id: int):
    if 'user_messages_to_delete' not in context.user_data:
        context.user_data['user_messages_to_delete'] = []
    context.user_data['user_messages_to_delete'].append(message_id)


# Создает клавиатуру из кнопок
def create_main_keyboard():
    """Клавиатура главного меню"""
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_TEXT_LENGTH), KeyboardButton(BTN_FILE_LENGTH)]],
        resize_keyboard=True
    )


# Инициализирует кнопку отмены
def create_cancel_keyboard():
    """Клавиатура с кнопкой отмены"""
    return ReplyKeyboardMarkup(
        [[KeyboardButton(BTN_CANCEL)]],
        resize_keyboard=True
    )


# Стартуем
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await delete_previous_messages(update, context)
    msg = await update.message.reply_text(
        "Выберите действие:",
        reply_markup=create_main_keyboard()
    )
    await track_message(context, msg.message_id)
    return MAIN_MENU


# Вызов предсказания категории
def predict_category(inpt):
    response = requests.post(API_ENDPOINT, json={"text": inpt})
    return response.json()


def send_mess_to_classification(inp):
    inp = inp.replace('\r\n', '')
    categories = predict_category(inp)['labels']
    categories = [cat['label'] + f" ({str(round(cat['score'], 2) * 100)}%)" for cat in categories]
    res = f"{inp}:\nКатегории:\n" + ', '.join(categories)
    return res


def parse_file_content(content):
    content = content.split(';')
    content = [i for i in content if len(i) >= 5]
    res_data = ""
    for itm in content:
        line = send_mess_to_classification(itm)

        res_data += line.replace('\r\n\t', '') + '\n\n'

    return res_data


# Задает кнопки главного меню (ввод текста, отправка файла)
# Реализует функционал по отправке результата обработки текста и содержимого файла
async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главное меню"""

    await delete_previous_messages(update, context)
    await track_user_message(context, update.message.message_id)

    text = update.message.text
    if text == BTN_TEXT_LENGTH:

        msg = await update.message.reply_text(
            "Введите новость:",
            reply_markup=create_cancel_keyboard()
        )
        await track_message(context, msg.message_id)

        return WAITING_TEXT

    elif text == BTN_FILE_LENGTH:

        msg = await update.message.reply_text(
            "Отправьте файл с новостями:",
            reply_markup=create_cancel_keyboard()
        )
        await track_message(context, msg.message_id)

        return WAITING_FILE
    return MAIN_MENU


# Обработка текстовых сообщений от пользователя
async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text == BTN_CANCEL:
        return await cancel(update, context)

    """Обработка введенного текста"""
    text = update.message.text

    await delete_previous_messages(update, context)

    await update.message.reply_text(
        f"{send_mess_to_classification(text)}",
        reply_markup=create_main_keyboard()
    )

    return MAIN_MENU


# Обработка отправленных файлов
async def handle_file_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text == BTN_CANCEL:
        return await cancel(update, context)

    """Обработка полученного файла"""
    temp_file_path = None
    try:
        # Получаем док и получаем его содержимое
        document = update.message.document
        file = await document.get_file()

        # Скачиваем содержимое в байты
        file_bytes = await file.download_as_bytearray()

        # Декодируем с нужной кодировкой
        content = file_bytes.decode('utf-8')

        # Новое содержимое файла                  
        new_content = parse_file_content(content)

        await delete_previous_messages(update, context)

        # Создаем файловый объект в памяти
        modified_file = io.BytesIO(new_content.encode('utf-8'))
        modified_file.name = f"modified_{document.file_name}"

        await update.message.reply_document(
            document=InputFile(modified_file),
            reply_markup=create_main_keyboard()
        )

    except Exception as e:
        logger.error(f"File error: {e}")
        await update.message.reply_text("❌ Ошибка обработки файла")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return MAIN_MENU


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Удаление сообщения с кнопкой "Отмена"
    try:
        await context.bot.delete_message(chat_id, update.message.message_id)
    except Exception as e:
        logger.error(f"Ошибка удаления: {e}")

    # Удаление системных сообщений
    await delete_previous_messages(update, context)

    # Удаление сообщений пользователя
    if 'user_messages_to_delete' in context.user_data:
        for msg_id in context.user_data['user_messages_to_delete']:
            try:
                await context.bot.delete_message(chat_id, msg_id)
            except Exception as e:
                logger.error(f"Ошибка удаления: {e}")
        context.user_data['user_messages_to_delete'] = []

    """Обработка отмены"""
    await delete_previous_messages(update, context)
    msg = await update.message.reply_text(
        "Действие отменено",
        reply_markup=create_main_keyboard()
    )

    await track_message(context, msg.message_id)
    return MAIN_MENU  # Важно: возвращаемся в главное меню


def main():
    """Настройка и запуск бота"""
    app = Application.builder().token(os.getenv("BOT_TOKEN")).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, main_menu)
            ],
            WAITING_TEXT: [
                MessageHandler(filters.Regex(f'^{BTN_CANCEL}$'), cancel),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input)
            ],
            WAITING_FILE: [
                MessageHandler(filters.Regex(f'^{BTN_CANCEL}$'), cancel),
                MessageHandler(document_filter, handle_file_input)
            ]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    app.add_handler(conv_handler)
    app.run_polling()


if __name__ == '__main__':
    main()
