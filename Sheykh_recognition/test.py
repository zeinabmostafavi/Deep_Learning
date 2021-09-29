import cv2
import telebot
import numpy as np
import tensorflow
from keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model("model.h5")
width = height = 230


bot = telebot.TeleBot("-----YOUR TOKEN-----")


@bot.message_handler(commands=['start'])
def say_hi(messages):
    bot.send_message(
        messages.chat.id, f'سلام {messages.from_user.first_name} عزیز😎 ')
    bot.send_message(
        messages.chat.id, f'سازنده من به من اموزش داده تا بتونم یک فرد شیخ و از فرد عادی شناسایی کنم 😋')
    bot.send_message(
        messages.chat.id, f' پس یک عکس واسم بفرس تا  بگم اون فرد شیخ  هست یا فرد عادی  ☺...')


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
    if np.argmax(pred) == 1:
        bot.reply_to(message, 'شیخ')
    elif np.argmax(pred) == 0:
        bot.reply_to(message, ' نرمال')


bot.polling()
