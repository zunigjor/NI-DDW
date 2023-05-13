# HW6 - Recommender systems

https://courses.fit.cvut.cz/NI-DDW/hw/06/index.html
___

Program pracuje nad education and development verzí datasetu [MovieLens](https://grouplens.org/datasets/movielens/).

Implementoval jsem funkci `get_user_profiles()`, která vytvoří uživatelské profily a vytvoří slovík obshující všechna hodnocení filmů uživateli. Tento slovník je později využit k vyfiltrování již hodnocených filmů.

Funkce `content_based()`, `collaboration_filtering()` a `hybrid()` vrací slovníky, kde klíčem je id uživatele a hodnotou je seznam filmů s jejich id, jménem a hodnotou v intervalu `[0 - 1]`, neboli podobnostím skore pro uživatele.

- Funkce `content_based()` pracuje nad vektory žánrů filmů a uživatelů a jejich podobnosti porovnává podle cosinové podobnosti.
- Funkce `collaboration_filtering()` pracuje nad vektory uživatelů a následně průměruje hodnocení pro filmy a z ních vypočítává podobnostní skore.
- Funkce `hybrid()` spojuje tyto dva výsledky s váhami: 
  - `content_based * 0.6 +  collaborative_filtering * 0.4`
  - Tyto váhy jsem vybral, převážně protože se mi osobně výsledky content_based zdály o něco málo lepší.

## Výsledky

Výsledky jsou v souboru `results/results.txt` a vypadají takto:
```
User 1
Top 5 content based:
    1: 0.864 117646 - Dragonheart 2: A New Beginning (2000) 
    2: 0.844  55116 - Hunting Party, The (2007) 
    3: 0.826   5657 - Flashback (1990) 
    4: 0.826   6990 - The Great Train Robbery (1978) 
    5: 0.825   4956 - Stunt Man, The (1980) 
Top 5 collaborative filtering:
    1: 0.578    318 - Shawshank Redemption, The (1994) 
    2: 0.552    589 - Terminator 2: Judgment Day (1991) 
    3: 0.542    588 - Aladdin (1992) 
    4: 0.538    364 - Lion King, The (1994) 
    5: 0.472   4306 - Shrek (2001) 
Top 5 hybrid:
    1: 0.588    156 - Blue in the Face (1995) 
    2: 0.576    653 - Dragonheart (1996) 
    3: 0.563    205 - Unstrung Heroes (1995) 
    4: 0.555   2002 - Lethal Weapon 3 (1992) 
    5: 0.554    327 - Tank Girl (1995) 
============================================================
User 2
...
```

Celkové zpracování trvalo poměrně dlouho, hlavně u collaboration filtering. Hlavní vylepšení co se mi podařilo, bylo ukládání výsledků pomocí knihovny `pickle`. To značne zrychlilo vývoj a debugging.

Co se týče dalších rozšíření tak by se dalo víc hrát s váhami u hybrid a případně i vylepšit výpis. Určitě by se daly optimalizovat některé výpočty aby probíhaly rychleji, to ale bude spíše problém jazyku python.
