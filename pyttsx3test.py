import pyttsx3

print("hello")

engine = pyttsx3.init()
voices = engine.getProperty('voices')

voices = [x.id for x in voices]

print(voices)
print("Hello")