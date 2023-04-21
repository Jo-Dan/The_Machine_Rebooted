"""
Voice recognition and random voice mp3 retreival (a la The Machine)

By Jo-dan
"""
# from gtts import gTTS
import os
import random
import string
import time

import natural.text
import pyaudio
import pocketsphinx
import pyttsx3
import speech_recognition as sr
from num2words import num2words
from pyglet import app, clock, media

cwd = os.getcwd()
beep = media.load("tts_downloads/beep.mp3", streaming=False)
languages = ['en-us', 'en-uk', ]

r = sr.Recognizer()

player = media.Player()


def exit_callback(dt):
    if player.playing != True:
        app.exit()


def get_mp3(text):
    text = text.replace(";", " beep ")
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = text.replace(",", "")

    lowertext = text.lower()
    words = lowertext.split()
    engine = pyttsx3.init()
    player.queue(beep)
    for inword in words:
        try:
            word = int(inword)
        except:
            word = inword
        save_path = cwd + '\\tts_downloads\\{}.mp3'.format(word)
        if os.path.isfile(save_path) == False:
            # tts = gTTS(word, 'en-us')
            # tts.save(save_path)
            voice = random.choice(engine.getProperty('voices'))
            engine.setProperty('voice', voice.id)
            engine.save_to_file(word, save_path)
            engine.runAndWait()
            # pvoice.voice_name = voicename
            # pvoice.fetch_voice(word, save_path)

        mp3 = media.load(save_path)
        player.queue(mp3)
    player.queue(beep)
    player.play()
    clock.schedule_interval(exit_callback, 1/10.0)
    app.run()
    print('\n' + text)


def get_speech():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print('\n>>>', end=' ')
        audio = r.listen(source)
    print(">", end=' ')
    try:
        # speech = r.recognize_google(audio)
        speech = r.recognize_sphinx(audio)
    except:
        speech = "No input detected."
    print(speech.replace(" beep ", ". "))
    return speech


replies = [['yes', 'yep', 'yeah', 'absolutely', 'definitely'],
           ['no', 'nope', 'nah'],
           ['goodbye', 'close', 'exit', 'quit', 'bye'],
           ['time'],
           ['hear me'],
           ['repeat'],
           ['nato'],
           ['text', 'input']]


def message_type(text):
    reply_type = ''
    for x in replies:
        for y in x:
            if y in text.lower():
                reply_type = x[0]
    if len(reply_type) == 0:
        return ""
    else:
        return reply_type


def get_nato(txt):
    nato_type = natural.text.code(txt, format='word')
    return str(nato_type)


def conversation(text):
    reply = message_type(text)
    # print '\n{}'.format(m_reply)
    # get_mp3(reply)
    if reply == 'goodbye':
        get_mp3('goodbye')
        global end
        end = True
    elif reply == 'time':
        word_hour = num2words(int(time.strftime("%I")))
        word_minute = num2words(
            int(time.strftime("%M").lstrip('0'))).replace('-', ' ')
        if word_minute == 'zero':
            word_minute = 'o clock'
        # word_minute = word_minute.split(' ')
        get_mp3('It is {} {}'.format(word_hour, word_minute))

    elif reply == 'repeat':
        get_mp3(text.lower().replace('repeat', ''))

    elif reply == 'nato':
        get_mp3(get_nato(text.lower().replace('nato', '')))

    elif reply == 'hear me':
        get_mp3('Yes')

    elif reply == 'text':
        txtin = input('>>>')
        conversation(txtin)

    else:
        get_mp3("I don't under stand")


# get_mp3('Can you hear me')
# hear = raw_input('Can you hear me? \n>>> ')
# hear = get_speech()
#
# if message_type(hear) == 'yes':
#    get_mp3('Good. what would you like to ask me')
#    end = False
#    while end == False:
#        # text = raw_input('>>> ')
#        text = get_speech()
#        conversation(text)
#
# else:
#    print 'Check your settings and try again.'
#
# print '\n######### \n Goodbye \n#########'
# time.sleep(5)
