from word_model import WordModel

words = ["wlacz", "wylacz", "start", "stop", "lewo", "prawo", "naprzod", "wstecz", "gora", "doł", "swiatlo", "wozek"]

for word in words:
    new_model = WordModel(word)
    new_model.load("/jakas_sciezka")
