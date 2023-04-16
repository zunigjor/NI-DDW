# HW2 - Text Mining

https://courses.fit.cvut.cz/NI-DDW/hw/02/index.html

___

* Jako vstup jsem použil články získané v [prvním domácím úkolu](../01).
* Výstupem je soubor [results/result.json](./results/result.json) ve formátu:
```json
[
  {
    "header": ...,
    "author": ...,
    "publish_date": ...,
    "body": ...,
    "POS_tagging": ...,
    "NER_entity_classification": ...,
    "NER_custom_pattern": ...,
    "NER_hugging_face": ...,
    "Wiki_classification": ...
  },
...
]
```
* Soubor obsahuje klasifikace podle POS, NER, custom NER, který hledá přídavná a podstatná jména, předtrénovaný NER bert od Hugging Face a NER v kombinaci s hledáním na wikipedii.
* POS klasifikace bohužel zpracovala i znaky jako ` ` nebo `“`, problémy dělaly i české symboly s háčky a čárky.
* Nejlépe hodnotím wiki klasifikaci, která poskytuje nejsrozumitelnější výsledky.
* Hugging Face Bert klasifikace mě příliš neoslnila, výsledky nltk NER považuji za lepší.
* Dalším rozšířením by mohla být detekce nálady článků. Jelikož se jedná o články vědecko-populární očekával bych převážně neutrální nebo jemně pozitivní náladu.
* Dále by se daly vyřešit problémy se speciálními znaky.

pozn.: Kód obsahuje `warnings.filterwarnings("ignore")` kvůli balíčku wikipedia, který hlásí warningy. Pro zachování hezčího výpisu do terminálu je ignoruji.
