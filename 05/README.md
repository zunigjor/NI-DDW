# HW5 - Indexing + Document Retrieval

https://courses.fit.cvut.cz/NI-DDW/hw/05/index.html
___

Program pracuje nad předzpracovaným datasetem [cranfield.zip](https://courses.fit.cvut.cz/NI-DDW/hw/05/files/cranfield.zip).

Program iteruje všechny query ve složce `data/q` a hledá relevantní dokumenty ve složce `data/r`. 
Pro toto hledání jsem implementoval čtyři způsoby vytvoření VSM (Vector Space Model) a získání seznamu relevantních dokumentů:
* **Binární** - funkce `Binary()`
* **TF** (Term Frequency) - funkce `TermFrequency()`
* **TF-IDF** (Term Frequency-Inverse Document Frequency) - funkce `TF_IDF()`
* Předtrénovaný model **distilbert-base-nli-stsb-mean-tokens** - funkce `distilbert()`

Pro každý z těchto způsobů jsem na základě **euklidovy vzdálenosti** a **cosinové podobnosti** vypočítal metriky **Precision**, **Recall** a **F-Measure**.

Výsledky se nachází v souboru `results/results.txt`.

## Vyhodnocení výsledků
Hodnocení jednotlivých výsledků nad každou query by bylo poměrně náročné a tak jsem se rozhodl raději spočítat střední hodnoty výsledků a hodnotit na základě toho.
```
============================================================
MEAN Binary:
  Euclidian:
    - Mean Precision:    0.00562962962962963
    - Mean Recall:       0.007555791207515346
    - Mean F-Measure:    0.006255733947436483
  Cosine:
    - Mean Precision:    0.11822222222222223
    - Mean Recall:       0.2603828845584086
    - Mean F-Measure:    0.152031669172059
------------------------------------------------------------
MEAN TF:
  Euclidian:
    - Mean Precision:    0.0044444444444444444
    - Mean Recall:       0.006672728506061839
    - Mean F-Measure:    0.0052056309726291935
  Cosine:
    - Mean Precision:    0.10577777777777776
    - Mean Recall:       0.22291151659430725
    - Mean F-Measure:    0.13389156754346915
------------------------------------------------------------
MEAN TF_IDF:
  Euclidian:
    - Mean Precision:    0.0017777777777777779
    - Mean Recall:       0.0016100332617574
    - Mean F-Measure:    0.0016524148203792253
  Cosine:
    - Mean Precision:    0.18814814814814815
    - Mean Recall:       0.3946670825615134
    - Mean F-Measure:    0.2380005093531993
------------------------------------------------------------
MEAN distilbert-base-nli-stsb-mean-tokens:
  Euclidian:
    - Mean Precision:    0.13155555555555556
    - Mean Recall:       0.20190036286804375
    - Mean F-Measure:    0.14690368949895619
  Cosine:
    - Mean Precision:    0.14533333333333334
    - Mean Recall:       0.22100965698924846
    - Mean F-Measure:    0.16149823642637684
------------------------------------------------------------
============================================================
```
Nejlepší výsledky poskytují metody **TF-IDF** a předtrénovaný **distilbert-base-nli-stsb-mean-tokens** a to převážně u cosínové podobnosti.

Nejnáročnější byla implementace předtrénovaného modelu, kdy jsem několik hodin bojoval se samotným spuštěním a crashi celého IDE. Předtrénovaný **distilbert-base-nli-stsb-mean-tokens** trval znatelně déle než ostatní  metody.

Další rozšíření by mohla zahrnout další předtrénované modely a využít i jiné statistické metody pro vyhodnocení.
