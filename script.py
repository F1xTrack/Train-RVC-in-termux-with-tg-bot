import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Параметры по умолчанию
LEARNING_RATE = 0.001
BATCH_SIZE = 1
EPOCHS = 5
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
selected_folder = None  # Переменная для хранения выбранной папки

# Токен бота
TOKEN = "7245200603:AAEYsKtM4a6OSdYC3QpsajIhM2YAqMeL4uc"
bot = telebot.TeleBot(TOKEN)

# Пример простой нейросетевой модели
class SimpleAudioModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleAudioModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Функция для извлечения Mel-спектрограммы из аудиофайла
def extract_features(file_path, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)  # Загрузка аудио
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Преобразование в dB
    return mel_spec_db

# Функция для подготовки данных
def load_data(audio_dir):
    features = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".ogg"):
            file_path = os.path.join(audio_dir, filename)
            mel_spec = extract_features(file_path)
            features.append(mel_spec)
    return np.array(features)

# Основная функция для обучения модели
def train_model(audio_dir, batch_size, epochs):
    print(f"Загрузка данных из {audio_dir}...")
    data = load_data(audio_dir)
    
    # Преобразуем данные для подачи в модель
    input_size = data.shape[1] * data.shape[2]  # Входной размер = высота * ширина спектрограммы
    data = data.reshape(-1, input_size)
    
    # Создаем модель
    model = SimpleAudioModel(input_size=input_size, hidden_size=256, output_size=10)  # output_size зависит от задачи
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.MSELoss()  # Например, для регрессии
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Начинаем тренировку
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(data), batch_size):
            inputs = torch.tensor(data[i:i + batch_size], dtype=torch.float32)
            labels = torch.randn(batch_size, 10)  # Генерация случайных меток, нужно заменить на реальные

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Эпоха {epoch+1}/{epochs}, Потери: {running_loss/len(data)}")

    print("Обучение завершено!")
    return model

# Стартовая команда /start
@bot.message_handler(commands=['start'])
def start(message):
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Запустить обучение", callback_data='start_training'))
    markup.add(InlineKeyboardButton("Указать количество эпох", callback_data='set_epochs'))
    markup.add(InlineKeyboardButton("Указать batch size", callback_data='set_batch_size'))
    markup.add(InlineKeyboardButton("Выбрать папку с ассетами", callback_data='set_assets_dir'))
    bot.send_message(message.chat.id, "Выберите действие:", reply_markup=markup)

# Обработка инлайн-кнопок
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    global selected_folder
    if call.data == 'start_training':
        bot.send_message(call.message.chat.id, "Начинаем обучение...")
        model = train_model(AUDIO_DIR, BATCH_SIZE, EPOCHS)
        bot.send_message(call.message.chat.id, "Обучение завершено.")
    elif call.data == 'set_epochs':
        bot.send_message(call.message.chat.id, "Введите количество эпох командой /epochs <количество>")
    elif call.data == 'set_batch_size':
        bot.send_message(call.message.chat.id, "Введите batch size командой /batch_size <размер>")
    elif call.data == 'set_assets_dir':
        show_folders(call.message)

# Команда для установки количества эпох
@bot.message_handler(commands=['epochs'])
def set_epochs(message):
    global EPOCHS
    try:
        EPOCHS = int(message.text.split()[1])
        bot.reply_to(message, f"Количество эпох установлено: {EPOCHS}")
    except (IndexError, ValueError):
        bot.reply_to(message, "Использование: /epochs <количество>")

# Команда для установки batch size
@bot.message_handler(commands=['batch_size'])
def set_batch_size(message):
    global BATCH_SIZE
    try:
        BATCH_SIZE = int(message.text.split()[1])
        bot.reply_to(message, f"Batch size установлен: {BATCH_SIZE}")
    except (IndexError, ValueError):
        bot.reply_to(message, "Использование: /batch_size <размер>")

# Отображение доступных папок
def show_folders(message):
    folders = [f for f in os.listdir(os.path.dirname(__file__)) if os.path.isdir(f)]
    markup = InlineKeyboardMarkup()
    for folder in folders:
        markup.add(InlineKeyboardButton(folder, callback_data=f'select_folder_{folder}'))
    bot.send_message(message.chat.id, "Выберите папку:", reply_markup=markup)

# Подтверждение выбора папки
@bot.callback_query_handler(func=lambda call: call.data.startswith('select_folder_'))
def select_folder(call):
    global selected_folder, AUDIO_DIR
    selected_folder = call.data.replace('select_folder_', '')
    AUDIO_DIR = os.path.join(os.path.dirname(__file__), selected_folder)
    markup = InlineKeyboardMarkup()
    markup.add(InlineKeyboardButton("Готово ✅", callback_data='confirm_selection'))
    bot.send_message(call.message.chat.id, f"Вы выбрали папку: {selected_folder}. Подтвердите выбор:", reply_markup=markup)

# Подтверждение выбора папки ассетов
@bot.callback_query_handler(func=lambda call: call.data == 'confirm_selection')
def confirm_selection(call):
    global AUDIO_DIR
    bot.send_message(call.message.chat.id, f"Папка с ассетами установлена: {AUDIO_DIR}")

# Запуск бота
bot.polling()
