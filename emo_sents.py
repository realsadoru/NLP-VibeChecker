#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Piotr Celiński"
__contact__ = "pgc@post.pl"
__credits__ = ["Piotr Celiński"]
__deprecated__ = False
__email__ = "pgc@post.pl"
__license__ = "MIT License"
__maintainer__ = "developer"
__status__ = "Development"
__version__ = "0.1.0"

from polemo_intensity.model_script import Model  # importing the custom model class
from transformers import AutoTokenizer, pipeline
import os
import torch

# tekst = """Wpis premiera Donalda Tuska dotyczący nowej amerykańskiej strategii bezpieczeństwa był w ostatnich 24 godzinach najczęściej wyświetlanym postem na świecie w tematyce społeczno-politycznej - podaje instytut analityczny Res Futura. Został wyświetlony ponad 21 milionów razy, częściej niż wpisy między innymi Viktora Orbana czy Elona Muska.
# "Drodzy amerykańscy przyjaciele, Europa jest waszym najbliższym sojusznikiem, a nie waszym problemem. I mamy wspólnych wrogów. Przynajmniej tak było przez ostatnie 80 lat. Musimy się tego trzymać, to jedyna rozsądna strategia naszego wspólnego bezpieczeństwa. Chyba że coś się zmieniło" - napisał w sobotę premier Tusk na platformie X. Szef rządu opublikował wpis w języku angielskim.
# """
# tekst = "Wpis premiera Donalda Tuska dotyczący nowej amerykańskiej strategii bezpieczeństwa był w ostatnich 24 godzinach najczęściej wyświetlanym postem na świecie w tematyce społeczno-politycznej - podaje instytut analityczny Res Futura."

# tekst = 'W odpowiedzi na wpis Rose\'a jeden z internautów napisał: "Głęboko ubolewam, że decyzja Prezydenta Karola Nawrockiego i jego Kancelarii położyła kres długoletniej tradycji obchodzenia Chanuki w Pałacu Prezydenckim". "Rozumiem, że jako Ambasador Stanów Zjednoczonych i wielki przyjaciel Polski, nie może Pan narzucać decyzji ani pełnić roli arbitra w wewnętrznych sprawach Polski. Niemniej jednak byłbym wdzięczny, gdyby za pośrednictwem Państwa kanałów komunikacji udało się przekazać sugestię skłaniającą do refleksji" - dodał.'

# tekst = """Dwa lata rządów Donalda Tuska miały być naprawą państwa, a stały się symbolem nieskuteczności i chaosu. Zamiast porządku w wymiarze sprawiedliwości – otwarta wojna instytucji i rządzenie uchwałami. Zamiast odpowiedzialnych finansów – rekordowe deficyty i dalsze zadłużanie kraju. Zamiast sprawnie działającej służby zdrowia – odwoływane zabiegi, szpitale w trybie awaryjnym i pacjenci zostawieni bez pomocy.
# Wielkie obietnice skończyły się na słowach. Nawet sztandarowe postulaty nie zostały zrealizowane, a gdy pojawiła się realna szansa na ich wprowadzenie, rządząca większość sama je odrzuciła. Miliony Polaków dostały nie zmianę, dostały to samo, co w latach 2007-2014. Wystarczyły dwa lata, by przypomnieć sobie, co znaczy Donald Tusk i jego rządy"""

tekst = """Dzisiejsze obchody Święta Marynarki Wojennej to moment szczególny – chwila, w której z dumą i wdzięcznością patrzymy na tych, którzy każdego dnia strzegą bezpieczeństwa Polski na morzu.
Marynarka Wojenna RP stoi dziś w centrum naszych działań na rzecz wzmacniania zdolności obronnych państwa. Inwestujemy w nowe okręty, systemy uzbrojenia i nowoczesne technologie, które pozwolą marynarzom jeszcze skuteczniej wykonywać swoje zadania.
Dziękuję wam Marynarze za odwagę, determinację i niezachwianą gotowość do działania w każdych warunkach. Dziękuję za służbę będącą świadectwem najwyższego profesjonalizmu oraz oddania ojczyźnie.
Nie ma bezpiecznej Polski i bezpiecznego Bałtyku bez polskich marynarzy"""


# tekst = """Ala ma kota, kot ma Alę, ona go kocha, on ją wcale."""

def polemo_intensity_analysis(tekst: str):
    model_directory = r"C:\Users\piotr\PycharmProjects\models-palyground\polemo_intensity"  # Your full path to the model's directory
    model = Model.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    inputs = tokenizer(tekst, return_tensors="pt")
    outputs = model(inputs['input_ids'], inputs['attention_mask'])

    # Print out the emotion ratings
    for emotion, rating in zip(['Happiness', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Pride', 'Valence', 'Arousal'],
                               outputs):
        print(f"{emotion}: {rating.item()}")


def huggingface_pipeline_analysis(tekst: str, model_path: str):
    use_cuda = torch.cuda.is_available()
    device_arg = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else None
    emo_pipe = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=device_arg,
        dtype=dtype,
        token=os.getenv('HF_TOKEN'),
        top_k=None,  # get all labels
    )
    results = emo_pipe(tekst)
    for result in results[0]:
        print(f"{result['label']}: {result['score']}")


def main():
    print("Polemo Intensity Model Analysis:")
    polemo_intensity_analysis(tekst)
    print("\nHuggingface Pipeline Analysis:")
    model_path = "poltextlab/xlm-roberta-large-pooled-polish-emotions9-v2"
    huggingface_pipeline_analysis(tekst, model_path=model_path)
    model_path = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
    huggingface_pipeline_analysis(tekst, model_path=model_path)
    model_path = "visegradmedia-emotion/Emotion_RoBERTa_polish6"
    print("\nHuggingface Pipeline Analysis:")
    huggingface_pipeline_analysis(tekst, model_path=model_path)
    model_path = "nie3e/plutchik-emotions-polish-poc"
    print("\nHuggingface Pipeline Analysis:")
    huggingface_pipeline_analysis(tekst, model_path=model_path)


if __name__ == "__main__":
    main()
