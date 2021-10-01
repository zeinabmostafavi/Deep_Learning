import cv2
import telebot
import numpy as np
import tensorflow
from keras.models import load_model
import os

model = load_model("model.h5")
width = height = 224


bot = telebot.TeleBot("--------Your token--------")


@bot.message_handler(commands=['start'])
def say_hi(messages):
    bot.send_message(
        messages.chat.id, f'Hi {messages.from_user.first_name}Dear😎 ')
    bot.send_message(
        messages.chat.id, f'My creator has trained me to be able to identify a iranian and an foreign person 😋')
    bot.send_message(
        messages.chat.id, f' Now send me your photo  ☺...')


@bot.message_handler(content_types=['photo'])
def photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = file_info.file_path
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    image = cv2.imread(src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = np.array(image)
    image = image/255
    image = image.reshape(1, width, height, 3)
    pred = model.predict(image)
    print('____________________________', pred)
    res = np.argmax(pred)
    if res == 0:
        bot.reply_to(message, 'IRANIAN 🇮🇷')
    elif res == 1:
        bot.reply_to(message, ' FOREIGN')


bot.polling()
