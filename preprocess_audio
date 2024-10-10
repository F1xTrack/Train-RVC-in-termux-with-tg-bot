import librosa
import numpy as np

# Функция предобработки аудиофайла
def preprocess_audio(audio_path, sr=22050):
    # Загружаем аудио, ресемплируем на частоту 22050 Гц
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Нормализуем аудиофайл
    audio = librosa.util.normalize(audio)
    
    # Преобразуем в мел-спектрограмму
    mel_spec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db
    
