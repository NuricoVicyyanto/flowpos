# bot telegram
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from random import randint
from io import BytesIO

# machine learning
from keras.models import load_model
import numpy as np
from keras.preprocessing import image


with open('token.txt', 'r') as f:
    TOKEN = f.read()

# telegram token
updater = Updater(TOKEN, use_context=True)

# function


def start(update, context):
    # bold
    update.message.reply_text(
        "*Selamat datang di Clasifibot tekan /info untuk informasi lebih lanjut*", parse_mode=telegram.ParseMode.MARKDOWN_V2)


def info(update, context):
    update.message.reply_text("""Perintah yang tersedia :
    - /start untuk memulai bot
    - /info  untuk mengetahui info bot

    kirim gambar dengan format jpg untuk mendeteksi jenis bunga, 
    Bot ini dapat mendeteksi 5 jenis bunga yaitu :
    - Tanaman Keladi(Caladium)
    - Tanaman Daun Bahagia(Dieffenbachia)
    - Tanaman Janda Sobek(Monstera)
    - Tanaman Oleander(Nerium Oleander)
    - Tanaman Lili perdamaian(spathipyllum)
    """)


def unknown(update, context):
    update.message.reply_text(
        "Maaf, perintah '%s' tidak dikenal" % update.message.text)


def save(update, context):
    chat_id = str(update.update_id)
    file_image = update.message.photo[0].get_file()
    f = BytesIO(file_image.download_as_bytearray())
    with open("images/"+chat_id+".jpg", "wb") as fi:
        fi.write(f.getbuffer())
    update.message.reply_photo(open("images/"+chat_id+".jpg", "rb"))

    # load machine learning model
    model = load_model('./ml/flowpos2.h5')

    # machine learning prediction
    test_image = image.load_img("images/"+chat_id+".jpg", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        update.message.reply_text("Diklasifikasikan sebagai Keladi(Caladium)")
    elif result[0][1] == 1:
        update.message.reply_text(
            "Diklasifikasikan sebagai Daun Bahagia(Dieffenbachia)")
    elif result[0][2] == 1:
        update.message.reply_text(
            "Diklasifikasikan sebagai Janda Sobek(Monstera)")
    elif result[0][3] == 1:
        update.message.reply_text(
            "Diklasifikasikan sebagai Oleander(Nerium Oleander)")
    elif result[0][4] == 1:
        update.message.reply_text(
            "Diklasifikasikan sebagai Lili perdamaian(spathipyllum)")
    else:
        update.message.reply_text("Gagal Diklasifikasi")


# command bot
updater.dispatcher.add_handler(CommandHandler("start", start))
updater.dispatcher.add_handler(CommandHandler("info", info))
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))

# add Message Handler
updater.dispatcher.add_handler(MessageHandler(Filters.photo, save))

# run bot
updater.start_polling()
updater.idle()
