import telebot
from telebot import types
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess_audio import preprocess_audio
from learn_model import SimpleRVCModel
import numpy as np
import os

# Токен бота
TOKEN = '7245200603:AAEYsKtM4a6OSdYC3QpsajIhM2YAqMeL4uc'

# Настройки по умолчанию
default_settings = {
    'experiment_name': 'experiment',
    'total_epochs': 500,
    'batch_size': 8,
    'audio_archive': None
}

# Инициализация бота
bot = telebot.TeleBot(TOKEN)

# Подключение к базе данных для хранения архивов
conn = sqlite3.connect('audio_files.db', check_same_thread=False)
cursor = conn.cursor()

# Создание таблицы для хранения информации об архивах
cursor.execute('''CREATE TABLE IF NOT EXISTS archives 
                  (user_id INTEGER, archive_name TEXT)''')
conn.commit()

# Переменная для хранения текущих настроек пользователя
user_settings = {}

# Класс для загрузки и обработки данных
class MyDataset(Dataset):
    def __init__(self, audio_archive):
        self.audio_files = []
        self.labels = []
        # Загрузка и предобработка аудиофайлов
        for file in audio_archive:
            mel_spec = preprocess_audio(file)
            self.audio_files.append(mel_spec)
            self.labels.append(0)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio = self.audio_files[idx]
        label = self.labels[idx]
        audio = torch.tensor(audio).unsqueeze(0)
        return audio, label

# Функция для старта процесса обучения
def start_training_process(chat_id, settings):
    # Загрузка данных
    dataset = MyDataset(settings['audio_archive'])
    loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True)

    # Создание модели
    model = SimpleRVCModel()
    device = torch.device('cpu')
    model.to(device)

    # Обучение модели
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(settings['total_epochs']):
        for i, (audio, label) in enumerate(loader):
            audio, label = audio.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(audio)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Сохранение модели
    torch.save(model.state_dict(),'model.pth')

    # Отправка модели пользователю
    bot.send_document(chat_id, open("model.pth", 'rb'))

# Команда /start
@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, "Привет! Нажми /train, чтобы начать обучение своей первой модели.")

# Команда /train
@bot.message_handler(commands=['train'])
def train_command(message):
    user_id = message.from_user.id

    # Если у пользователя ещё нет настроек, добавим их
    if user_id not in user_settings:
        user_settings[user_id] = default_settings.copy()

    # Создаем инлайн-кнопки
    markup = types.InlineKeyboardMarkup(row_width=3)
    button1 = types.InlineKeyboardButton(f"Модель: {user_settings[user_id]['experiment_name']}", callback_data='change_name')
    button2 = types.InlineKeyboardButton(f"Эпохи: {user_settings[user_id]['total_epochs']}", callback_data='change_epochs')
    button3 = types.InlineKeyboardButton(f"Батч: {user_settings[user_id]['batch_size']}", callback_data='change_batch')
    button4 = types.InlineKeyboardButton(f"Аудиофайлы: {user_settings[user_id]['audio_archive'] or 'Загрузить'}", callback_data='change_audio')
    button5 = types.InlineKeyboardButton("Начать ", callback_data='start_training')

    markup.add(button1, button2, button3)
    markup.add(button4)
    markup.add(button5)

    bot.send_message(message.chat.id, "Настройки обучения:", reply_markup=markup)

# Обработка нажатий инлайн-кнопок
@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    user_id = call.from_user.id

    # Изменение названия модели
    if call.data == 'change_name':
        msg = bot.send_message(call.message.chat.id, "Введите новое название модели:")
        bot.register_next_step_handler(msg, set_experiment_name)

    # Изменение количества эпох
    elif call.data == 'change_epochs':
        msg = bot.send_message(call.message.chat.id, "Введите количество эпох:")
        bot.register_next_step_handler(msg, set_total_epochs)

    # Изменение размера батча
    elif call.data == 'change_batch':
        msg = bot.send_message(call.message.chat.id, "Введите размер батча:")
        bot.register_next_step_handler(msg, set_batch_size)

    # Изменение аудиофайлов
    elif call.data == 'change_audio':
        cursor.execute('SELECT archive_name FROM archives WHERE user_id=?', (user_id,))
        archives = cursor.fetchall()
        markup = types.InlineKeyboardMarkup(row_width=1)
        for archive in archives:
            markup.add(types.InlineKeyboardButton(archive[0], callback_data=f'select_audio_{archive[0]}'))
        bot.send_message(call.message.chat.id, "Выберите архив:", reply_markup=markup)

    # Начало обучения
    elif call.data =='start_training':
        bot.send_message(call.message.chat.id, "Начинаем обучение модели...")
        start_training_process(call.message.chat.id, user_settings[user_id])

# Функции изменения параметров
def set_experiment_name(message):
    user_settings[message.from_user.id]['experiment_name'] = message.text
    train_command(message)

def set_total_epochs(message):
    try:
        user_settings[message.from_user.id]['total_epochs'] = int(message.text)
    except ValueError:
        bot.send_message(message.chat.id, "Неправильный формат! Введите число.")
    train_command(message)

def set_batch_size(message):
    try:
        user_settings[message.from_user.id]['batch_size'] = int(message.text)
    except ValueError:
        bot.send_message(message.chat.id, "Неправильный формат! Введите число.")
    train_command(message)

# Команда /upload_audio
@bot.message_handler(commands=['upload_audio'])
def upload_audio(message):
    msg = bot.send_message(message.chat.id, "Загрузите архив с аудиофайлами.")
    bot.register_next_step_handler(msg, receive_audio_archive)

def receive_audio_archive(message):
    if message.document:
        file_name = message.document.file_name
        cursor.execute('INSERT INTO archives (user_id, archive_name) VALUES (?,?)', (message.from_user.id, file_name))
        conn.commit()
        
        # Проверяем, существует ли ключ в словаре user_settings
        if message.from_user.id not in user_settings:
            user_settings[message.from_user.id] = default_settings.copy()
        
        user_settings[message.from_user.id]['audio_archive'] = file_name
        bot.send_message(message.chat.id, f"Архив {file_name} загружен.")
        # Обрабатываем аудиофайлы
        audio_files = []
        for file in os.listdir('audio_files'):
            if file.endswith('.wav'):
                audio_files.append(os.path.join('audio_files', file))
        # Предобработка аудиофайлов
        mel_specs = []
        for audio_file in audio_files:
            mel_spec = preprocess_audio(audio_file)
            mel_specs.append(mel_spec)
        # Сохраняем предобработанные аудиофайлы
        np.save('preprocessed_audio.npy', mel_specs)
        train_command(message)

# Запуск бота
bot.polling(none_stop=True)
