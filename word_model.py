def resampling(x, fs1, fs2):
    out = x
    return out


class WordModel:
    name: str

    # jakaś struktura modelu

    def __init__(self, name: str):
        self.name = name

    def save(self, path: str):
        return ValueError

    def load(self, path: str):
        return ValueError

    def train(self, path: str):
        # trzeba zrobić jakiś model danego słowa (nie wiem, HMM, za bardzo nie pamiętam jak to się robiło)
        # 1) Wczytanie wszystkich sygnałów danego słowa
        # 2) Resampling do 8kHz
        # 3) Ramkowanie (25-30ms + nakładka 25%)
        # 4) Parametryzacja (26 elementów -> 13 elementów (energia+12MFCC) + delta
        # 5) Stworzenie modelu HMM
        return ValueError
