#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

from transformers import AutoTokenizer, pipeline
from polemo_intensity.model_script import Model


# wizualizacja wyników
def draw_bar(score, max_val=1.0, width=20):
    # prosty pasek tekstowy
    # Polemo czasem wypluwa wartości > 1.0, więc lekko to przycinam
    safe_score = max(0, min(score, max_val))

    filled = int((safe_score / max_val) * width)
    empty = width - filled

    bar = "█" * filled + "░" * empty
    return f"|{bar}| {score:.4f}"


def load_sentences_from_file(path):
    # czytamy plik linia po linii
    collected = []

    if not os.path.exists(path):
        print(f"Plik {path} nie istnieje.")
        return collected

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line.startswith("-"):
                # usuwamy myślnik i ewentualne spacje
                collected.append(line[1:].strip())

    return collected


def polemo_intensity_analysis(sentences):
    # ścieżka na sztywno
    model_dir = "/Users/maciej/PycharmProjects/emo_sents/polemo_intensity"

    model = Model.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    emotion_labels = [
        "Happiness",
        "Sadness",
        "Anger",
        "Disgust",
        "Fear",
        "Pride",
        "Valence",
        "Arousal",
    ]

    results_buffer = []

    for tekst in sentences:
        tokens = tokenizer(
            tekst,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output = model(tokens["input_ids"], tokens["attention_mask"])

        # rozpakowujemy tensory do floatów
        results_buffer.append([val.item() for val in output])

    if not results_buffer:
        print("Brak wyników z Polemo (to raczej nie powinno się zdarzyć).")
        return

    avg_vals = np.mean(results_buffer, axis=0)
    results_map = dict(zip(emotion_labels, avg_vals))

    # sortujemy od największej wartości
    sorted_results = sorted(results_map.items(), key=lambda item: item[1], reverse=True)

    print("\n" + "-" * 50)
    print(" Model: polemo_intensity")
    print("-" * 50)

    for label, val in sorted_results:
        print(f"{label:12} {draw_bar(val)}")


def huggingface_pipeline_analysis(sentences, model_name):
    # apple MPS zamiast CUDA
    device = "mps" if torch.backends.mps.is_available() else -1

    emo_pipe = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
        token=os.getenv("HF_TOKEN"),
        top_k=None,
    )

    summed_scores = {}
    total = len(sentences)

    for tekst in sentences:
        # pipeline zwraca listę, w której pierwszy element zawiera wszystkie etykiety
        output = emo_pipe(tekst, truncation=True, max_length=512)

        for row in output[0]:
            lbl = row["label"]
            sc = row["score"]

            # klasyczny pattern zliczania bez defaultdict
            if lbl not in summed_scores:
                summed_scores[lbl] = 0.0
            summed_scores[lbl] += sc

    # średnie wartości
    avg_scores = {k: v / total for k, v in summed_scores.items()}
    avg_sorted = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "-" * 50)
    print(f" Model: {model_name}")
    print("-" * 50)

    for label, score in avg_sorted:
        print(f"{label:15} {draw_bar(score)}")


def print_welcome_message():
    # cool napis
    banner = """
       ███╗   ██╗██╗     ██████╗     ██╗   ██╗██╗██████╗ ███████╗
       ████╗  ██║██║     ██╔══██╗    ██║   ██║██║██╔══██╗██╔════╝
       ██╔██╗ ██║██║     ██████╔╝    ██║   ██║██║██████╔╝█████╗  
       ██║╚██╗██║██║     ██╔═══╝     ╚██╗ ██╔╝██║██╔══██╗██╔══╝  
       ██║ ╚████║███████╗██║          ╚████╔╝ ██║██████╔╝███████╗
       ╚═╝  ╚═══╝╚══════╝╚═╝           ╚═══╝  ╚═╝╚═════╝ ╚══════╝

             ██████╗██╗  ██╗███████╗ ██████╗██╗  ██╗███████╗██████╗ 
            ██╔════╝██║  ██║██╔════╝██╔════╝██║ ██╔╝██╔════╝██╔══██╗
            ██║     ███████║█████╗  ██║     █████╔╝ █████╗  ██████╔╝
            ██║     ██╔══██║██╔══╝  ██║     ██╔═██╗ ██╔══╝  ██╔══██╗
            ╚██████╗██║  ██║███████╗╚██████╗██║  ██╗███████╗██║  ██║
             ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
       """
    print(banner)


def main():
    print_welcome_message()

    input_file = "artykuły.txt"
    sentences = load_sentences_from_file(input_file)

    if not sentences:
        print("Nie znaleziono żadnych zdań do analizy.")
        return

    print(f"Rozpoczynam analizę ({len(sentences)} zdań)...")

    # najpierw Polemo
    polemo_intensity_analysis(sentences)

    # potem modele z HuggingFace
    hf_models = [
        "poltextlab/xlm-roberta-large-pooled-polish-emotions9-v2",
        "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned",
        "visegradmedia-emotion/Emotion_RoBERTa_polish6",
        "nie3e/plutchik-emotions-polish-poc",
    ]

    for model_name in hf_models:
        huggingface_pipeline_analysis(sentences, model_name)

    print()
    print("Analiza zakończona pomyślnie :)" )


if __name__ == "__main__":
    main()
