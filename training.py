from word_model import WordModel

words = ["wlacz", "wylacz", "start", "stop", "lewo", "prawo", "naprzod", "wstecz", "gora", "dol", "swiatlo", "wozek"]

for word in words:
    new_model = WordModel(word)
    new_path = "./Komendy/" + word + "/"
    new_model.train(new_path)
