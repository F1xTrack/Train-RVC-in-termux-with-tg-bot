import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler

# Параметры по умолчанию
LEARNING_RATE = 0.001
BATCH_SIZE = 1
EPOCHS = 5
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "assets")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Токен бота
TOKEN = "7245200603:AAEYsKtM4a6OSdYC3QpsajIhM2YAqMeL4uc"

# Глобальная переменная для хранения выбора папки
selected_folder = None

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

# Создание инлайн-кнопок для управления параметрами
def start(update, context):
    keyboard = [
        [InlineKeyboardButton("Запустить обучение", callback_data='start_training')],
        [InlineKeyboardButton("Указать количество эпох", callback_data='set_epochs')],
        [InlineKeyboardButton("Указать batch size", callback_data='set_batch_size')],
        [InlineKeyboardButton("Выбрать папку с ассетами", callback_data='set_assets_dir')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Выберите действие:', reply_markup=reply_markup)

# Обработка нажатий инлайн-кнопок
def button(update, context):
    global selected_folder
    query = update.callback_query
    query.answer()
    
    if query.data == 'start_training':
        query.edit_message_text(text="Начинаем обучение...")
        # Запуск обучения с текущими параметрами
        model = train_model(AUDIO_DIR, BATCH_SIZE, EPOCHS)
        query.edit_message_text(text="Обучение завершено.")
    elif query.data == 'set_epochs':
        query.edit_message_text(text="Введите количество эпох командой /epochs <количество>")
    elif query.data == 'set_batch_size':
        query.edit_message_text(text="Введите batch size командой /batch_size <размер>")
    elif query.data == 'set_assets_dir':
        show_folders(update, context)
    elif query.data.startswith('select_folder_'):
        selected_folder = query.data.replace('select_folder_', '')
        query.edit_message_text(text=f"Вы выбрали папку: {selected_folder}")
        show_confirm_button(update, context)

# Функция для показа кнопки подтверждения выбора папки
def show_confirm_button(update, context):
    keyboard = [[InlineKeyboardButton("Готово ✅", callback_data='confirm_selection')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=update.callback_query.message.chat_id, text="Подтвердите выбор:", reply_markup=reply_markup)

# Функция для отображения списка папок
def show_folders(update, context):
    folders = [f for f in os.listdir(os.path.dirname(__file__)) if os.path.isdir(f)]
    keyboard = [[InlineKeyboardButton(folder, callback_data=f'select_folder_{folder}')] for folder in folders]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.callback_query.edit_message_text(text="Выберите папку:", reply_markup=reply_markup)

# Подтверждение выбора папки
def confirm_selection(update, context):
    global AUDIO_DIR
    if selected_folder:
        AUDIO_DIR = os.path.join(os.path.dirname(__file__), selected_folder)
        context.bot.send_message(chat_id=update.callback_query.message.chat_id, text=f"Папка с ассетами установлена: {AUDIO_DIR}")
    else:
        context.bot.send_message(chat_id=update.callback_query.message.chat_id, text="Выбор папки не был сделан.")

# Установка количества эпох через команду
def set_epochs(update, context):
    global EPOCHS
    try:
        EPOCHS = int(context.args[0])
        update.message.reply_text(f"Количество эпох установлено: {EPOCHS}")
    except (IndexError, ValueError):
        update.message.reply_text("Использование: /epochs <количество>")

# Установка batch size через команду
def set_batch_size(update, context):
    global BATCH_SIZE
    try:
        BATCH_SIZE = int(context.args[0])
        update.message.reply_text(f"Batch size установлен: {BATCH_SIZE}")
    except (IndexError, ValueError):
        update.message.reply_text("Использование: /batch_size <размер>")

def main():
    # Настройка бота
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # Команды бота
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("epochs", set_epochs))
    dp.add_handler(CommandHandler("batch_size", set_batch_size))

    # Обработка нажатий кнопок
    dp.add_handler(CallbackQueryHandler(button))
    dp.add_handler(CallbackQueryHandler(confirm_selection, pattern='confirm_selection'))

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
