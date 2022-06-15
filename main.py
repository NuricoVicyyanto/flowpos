# bot telegram
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from random import randint
from io import BytesIO
import openai
import os

# machine learning
from keras.models import load_model
import numpy as np
from keras.preprocessing import image


with open('token.txt', 'r') as f:
    TOKEN = f.read()

# with open('aiToken.txt', 'r') as f:
#     AI_TOKEN = f.read()

# openai.api_key = AI_TOKEN

# telegram token
updater = Updater(TOKEN, use_context=True)


def start(update, context):
    # bold
    update.message.reply_text("""
    *Selamat datang di FlowBot*
    FlowBot merupakan bot yang dapat membantu anda mengenali tanaman hias beracun, tekan /info untuk informasi lebih lanjut""", parse_mode=telegram.ParseMode.MARKDOWN_V2)


def info(update, context):
    update.message.reply_text("""Selamat datang di FlowBot

    Perintah yang tersedia :
    - /start untuk memulai bot
    - /info  untuk informasi mengenai bot
    - /show untuk menampilkan contoh gambar tanaman beracun

    Panduan :
    kirim gambar dengan format jpg dengan file terkompresi untuk 
    mendeteksi jenis tanaman hias,

    Bot ini dapat mendeteksi 5 jenis tanaman yaitu :
    - Tanaman Keladi(Caladium)
    - Tanaman Daun Bahagia(Dieffenbachia)
    - Tanaman Janda Sobek(Monstera)
    - Tanaman Oleander(Nerium Oleander)
    - Tanaman Lili perdamaian(spathipyllum)

    Sumber :
    - https://libguides.nybg.org/poisonoushouseplants
    """)


def show(update, context):
    update.message.reply_photo(open('data/caladium.jpg', 'rb'))
    update.message.reply_text("Tanaman Keladi(Caladium)")
    update.message.reply_photo(open('data/dieffenbachia.jpg', 'rb'))
    update.message.reply_text("Tanaman Daun Bahagia(Dieffenbachia)")
    update.message.reply_photo(open('data/monstera.jpg', 'rb'))
    update.message.reply_text("Tanaman Janda Sobek(Monstera)")
    update.message.reply_photo(open('data/oleander.jpg', 'rb'))
    update.message.reply_text("Tanaman Oleander(Nerium Oleander)")
    update.message.reply_photo(open('data/spathipyllum.jpg', 'rb'))
    update.message.reply_text("Tanaman Lili perdamaian(spathipyllum)")


def save(update, context):
    chat_id = str(update.update_id)
    file_image = update.message.photo[0].get_file()
    f = BytesIO(file_image.download_as_bytearray())
    with open("images/"+chat_id+".jpg", "wb") as fi:
        fi.write(f.getbuffer())
    update.message.reply_photo(open("images/"+chat_id+".jpg", "rb"))

    # load machine learning model
    model = load_model('./model/flowpos2.h5')

    # machine learning prediction
    test_image = image.load_img("images/"+chat_id+".jpg", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] >= 0.8:
        update.message.reply_text("*Klasifikasi* \nTanaman diklasifikasikan sebagai Keladi, _Caladium_ \n\n*Bagian beracun* \nSemua bagian tanaman mengandung kalsium oksalat \n\n*Gejala yang ditimbulkan* \nIritasi parah pada selaput lendir menghasilkan pembengkakan lidah, bibir dan langitlangit", parse_mode=telegram.ParseMode.MARKDOWN_V2)
        update.message.reply_audio(open('audio/caladium.mp3', 'rb'))
    elif result[0][1] >= 0.8:
        update.message.reply_text("*Klasifikasi* \nTanaman diklasifikasikan sebagai Daun Bahagia, _Dieffenbachia_ \n\n*Bagian beracun* \nSemua bagian tanaman mengandung kalsium oksalat \n\n*Gejala yang ditimbulkan* \nIritasi parah pada selaput lendir menghasilkan pembengkakan lidah, bibir dan langitlangit", parse_mode=telegram.ParseMode.MARKDOWN_V2)
    elif result[0][2] >= 0.8:
        update.message.reply_text("*Klasifikasi* \nTanaman diklasifikasikan sebagai Janda Sobek, _Monstera_ \n\n*Bagian beracun* \nDaunnya mengandung kalsium oksalat \n\n*Gejala yang ditimbulkan* \nIritasi parah pada selaput lendir menghasilkan pembengkakan lidah, bibir dan langitlangit", parse_mode=telegram.ParseMode.MARKDOWN_V2)
    elif result[0][3] >= 0.8:
        update.message.reply_text("*Klasifikasi* \nTanaman diklasifikasikan sebagai Oleander, _Nerum Oleander_ \n\n*Bagian beracun* \nSemua bagian mengandung glikosida \n\n*Gejala yang ditimbulkan* \nSatu daun berakibat fatal dan akan mengganggu fungsi jantung, memicu kegagalan peredaran darah dan menyebabkan kematian", parse_mode=telegram.ParseMode.MARKDOWN_V2)
    elif result[0][4] >= 0.8:
        update.message.reply_text("*Klasifikasi* \nTanaman diklasifikasikan sebagai Lily Perdamaian, _Spathipyllum_ \n\n*Bagian beracun* \nDaunnya mengandung kalsium oksalat \n\n*Gejala yang ditimbulkan* \nIritasi parah pada selaput lendir menghasilkan pembengkakan lidah, bibir dan langitlangit", parse_mode=telegram.ParseMode.MARKDOWN_V2)
    else:
        update.message.reply_text("Gagal diklasifikasi")


def unknown(update, context):
    input = update.message.reply_text(update.message.text)
    update.message.reply_text(
        "Maaf, perintah '%s' tidak dikenali" % update.message.text)


# experimental
# def aiChat(update, context):
#     response = openai.Completion.create(
#         engine="text-ada-001",
#         prompt='"""\n{}\n"""'.format(update.message.text),
#         temperature=0,
#         max_tokens=1200,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=['"""'])

#     update.message.reply_text(response.choices[0].text)


# command bot
updater.dispatcher.add_handler(CommandHandler("start", start))
updater.dispatcher.add_handler(CommandHandler("info", info))
updater.dispatcher.add_handler(CommandHandler("show", show))
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))

# experimental
# updater.dispatcher.add_handler(MessageHandler(Filters.text, aiChat))

# add Message Handler
updater.dispatcher.add_handler(MessageHandler(Filters.photo, save))

# run bot
updater.start_polling()
updater.idle()
