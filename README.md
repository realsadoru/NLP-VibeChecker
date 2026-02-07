# Analiza wybranych tekstów pod kątem treści (rodzaju), walencji i natężenia emocji w odniesieniu do rozpoznanych nazwanych bytów (Named Entities).

<h2>Dobór danych i ekstrakcja treści</h2>

Podmiotem poddanym analizie (wybranym bytem) jest Władimir Putin. Materiał badawczy został pozyskany z portalu polsatnews.pl. W celu odiltrowania artykułów, w ktorych byt wzmiankowany jest tylko raz zastosowałem technikę Google Dorking przy użyciu operatora `site:polsatnews.pl "Putin" "Putin"`. 

Proces przygotowania danych przebiegał następująco:
1. Selekcja manualna: Z odnalezionych artykułów wyodrębniono zdania bezpośrednio wzmiankujące badany byt lub bezpośrednio go dotyczące.
2. Strukturyzacja danych: Wyekstraktowane treści zostały zapisane w pliku `artykuły.txt` z zachowaniem określonej struktury składniowej (każde zdanie poprzedzone myślnikiem).
3. Integracja ze skryptem: Zastosowany format danych jest ściśle powiązany z funkcją `load_sentences_from_file`, która wykorzystuje znacznik myślnika jako separator do automatycznego pomijania nagłówków i metadanych, skupiając proces analizy wyłącznie na wyselekcjonowanej warstwie tekstowej.

<h2>Metodologia i rozwój narzędzia</h2>

Punktem wyjścia dla opracowanego narzędzia był skrypt przygotowany przez prowadzącego (`emo_sents.py`). W trakcie pracy skrypt ten został wielu modyfikacjom, co doprowadziło do powstania finalnej wersji narzędzia (plik `nlpvibechecker.py`).

Główne różnice i wprowadzone usprawnienia:
1. Adaptacja środowiska i debugowanie: W skrypcie bazowym ścieżki do modeli oraz parametry sprzętowe były skonfigurowane pod konkretne środowisko lokalne (ścieżki w systemie Windows, wymuszenie CUDA). W wersji main.py zaimplementowano automatyczne wykrywanie sprzętu, ze szczególnym uwzględnieniem akceleracji Apple Silicon (MPS), co pozwoliło na natywne i wydajne działanie skryptu na systemach macOS.
2. Automatyzacja procesowania danych (Batch Processing): Pierwotny skrypt analizował pojedynczą zmienną tekstową zdefiniowaną "na sztywno" w kodzie. Wprowadzono funkcję `load_sentences_from_file`, która umożliwia dynamiczne wczytywanie danych z zewnętrznego pliku (`artykuły.txt`). Skrypt został wzbogacony o logikę selektywnego wyboru zdań (filtrowanie linii zaczynających się od myślnika).
3. Agregacja i analiza statystyczna: Zamiast jednostkowych wyników dla jednego fragmentu tekstu, narzędzie przetwarza cały zbiór danych (w testowanym przypadku 119 zdań), a następnie oblicza średnie wartości natężenia emocji dla całego korpusu przy użyciu biblioteki `numpy`. Dzięki temu wyniki są reprezentatywne dla całego zbioru artykułów, a nie tylko pojedynczych wypowiedzi.
4. Wizualizacja danych: Surowy wydruk liczbowy z wersji bazowej został zastąpiony autorską funkcją wizualizacyjną draw_bar. Generuje ona w konsoli graficzne paski postępu (ASCII bar charts), co pozwala na szybką i wzrokową ocenę profilu emocjonalnego bez konieczności analizy danych liczbowych. Dodatkowo wyniki są automatycznie sortowane od najwyższego natężenia, co ułatwia identyfikację dominujących nastrojów.
5. Struktura kodu: Kod został podzielony na modularne funkcje z wyraźnym oddzieleniem logiki ładowania danych, analizy specyficznym modelem PolEmo oraz obsługi pipelines HuggingFace.

<h2>Raport z analizy sentymentu i detekcji emocji</h2>

<h3>Metodologia badawcza</h3>

Analizę przeprowadzono wykorzystując pięć zróżnicowanych modeli deep learning:
1. PolEmo (hplisiecki)
2. XLM-RoBERTa (Poltextlab)
3. Twitter XLM-RoBERTa (Citizenlab)
4. Emotion RoBERTa (VisegradMedia)
5. Plutchik Emotions (nie3e)

<h3>Wyniki analizy i ich interpretacja</h3>

Poniżej znajdują się zrzuty ekranu wyników dla każdego modelu pochodzące bezpośrednio z terminala.

<img width="410" height="241" alt="image" src="https://github.com/user-attachments/assets/4c9cc4ed-88e7-4eb8-b05d-8c6bc1ffb958" />

<img width="510" height="269" alt="image" src="https://github.com/user-attachments/assets/dbe3e088-2859-4f74-9056-d7e37e632c18" />

<img width="518" height="132" alt="image" src="https://github.com/user-attachments/assets/9dedde8b-3785-4808-ae30-66a5daeb0b8d" />

<img width="434" height="197" alt="image" src="https://github.com/user-attachments/assets/276a5662-0f72-4760-9100-c936b5b70067" />

<img width="407" height="244" alt="image" src="https://github.com/user-attachments/assets/191e38db-811b-4845-8b93-bbe4cdf78ac8" />

<h4>Dominacja narracji obiektywnej</h4>

Kluczowym wnioskiem jest przewaga tonu neutralnego w badanym korpusie. Model Citizenlab zaklasyfikował aż *94,36%* treści jako Neutralne. Potwierdza to, że analizowane dane mają charakter sprawozdawczy (np. relacje z rozmów telefonicznych z premierem Izraela czy prezydentem Iranu).

<h4>Profil emocjonalny i składowe afektywne</h4>

Mimo neutralnej formy, modele detekcji emocji wykazują obecność silnych markerów afektywnych:
- Model oparty na kole Plutchika wskazał Anger (0.2633) jako dominującą emocję. Jest to skorelowane z obecnością w tekstach gróźb militarnych (np. "rozwiązanie zadań drogą zbrojną") oraz agresywnej retoryki politycznej ("młode świnie").
- Model Polemo wskazał wartość 0.4976 dla Valence. Relatywnie wysoki wynik (bliski środka skali) wynika z koegzystencji komunikatów o "pomocy" i "pokoju" obok opisów konfliktów.

<h4>Specyfika modeli a interpretacja kontekstu</h4>
Modele odnotowały obecność kategorii Hope (0.0899) i Trust (0.1728). Prawdopodobnie są one efektem występowania w tekstach słownictwa związanego z negocjacjami na Alasce, gwarancjami bezpieczeństwa oraz listem Melanii Trump dotyczącym ochrony dzieci. Niski wskaźnik pobudzenia (0.2423) sugeruje, że analizowane newsy, mimo drastycznej tematyki (np. pobór do wojska na rok 2026), są formułowane w sposób statyczny i sformalizowany.

<h3>Wnioski końcowe</h3>

Analiza wykazała zjawisko "agresywnej treści w neutralnej formie". Systemy AI poprawnie zidentyfikowały, że badany zbiór danych składa się z faktograficznych opisów zdarzeń o wysokim ładunku negatywnym. Rozbieżności między modelami (np. dominacja neutralności w jednym i gniewu w drugim) wynikają z różnych zbiorów treningowych – modele "twitterowe" silniej odfiltrowują suchy fakt od emocji, podczas gdy modele oparte na teorii Plutchika głębiej analizują semantykę użytych czasowników i rzeczowników.
