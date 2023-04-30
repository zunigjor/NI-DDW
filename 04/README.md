# HW4 - Web Analytics/Web Usage Mining

https://courses.fit.cvut.cz/NI-DDW/hw/04/index.html

---

Vstupem programu je dataset [wum-dataset-hw.zip](https://courses.fit.cvut.cz/NI-DDW/hw/04/files/wum-dataset-hw.zip). 


## Statistiky

Pro práci s daty byla využita knihovna pandas. Knihovna obsahuje i funkci `describe()`, která slouží k popisu dataframů načtených z csv souborů a jejich statistik. 

```
======================== CLICKS DESCRIPTION ========================
             LocalID        PageID  ...     PageScore  SequenceNumber
count   38451.000000  38451.000000  ...  38451.000000    38451.000000
mean   667685.000000   3286.412551  ...    143.092975        3.555122
std     11099.991937    408.577090  ...    260.595877        4.269960
min    648460.000000   3044.000000  ...     30.000000        1.000000
25%    658072.500000   3047.000000  ...     30.000000        1.000000
50%    667685.000000   3093.000000  ...     62.000000        2.000000
75%    677297.500000   3317.000000  ...    125.000000        4.000000
max    686910.000000   5258.000000  ...   5753.000000       50.000000

[8 rows x 9 columns]
================== SEARCH ENGINE MAP DESCRIPTION  ==================
       Referrer   Type
count       140    134
unique      140      5
top       URI_0  Other
freq          1     73
======================= VISITORS DESCRIPTION  ======================
            VisitID          Hour  Length_seconds  Length_pagecount
count  15559.000000  15559.000000    15559.000000      15559.000000
mean    9225.450157     13.814705      128.908028          2.471239
std     4654.823196      4.809969      328.777507          2.998959
min     1185.000000      0.000000        0.000000          1.000000
25%     5181.500000     11.000000        0.000000          1.000000
50%     9223.000000     14.000000        0.000000          1.000000
75%    13266.500000     17.000000      120.000000          3.000000
max    17265.000000     23.000000     5280.000000         50.000000
```

## Zpracování dat

Dataset jsem pročistil několika způsoby:
* V dataframu `clicks` jsem zredukoval TimeOnPage pomocí funkce `cut()` na intervaly.  
* Odstranil jsem sloupec `LocalID`, který neměl pro tento úkol význam.
* Odstranil jsem záznamy, kde čas návštěvy stránky byl pod 30 vteřin. 

Po těchto krocích se velikost vstupu zmenšila zhruba o 11400 záznamů.

```
=========================== MERGED SHAPE ===========================
38451 rows x 18 columns
================== MERGED WITHOUT SHORT VISITS SHAPE ===============
27041 rows x 18 columns
```

## Main a Micro konverze

Pro vytvoření datasetu pro algoritmus apriory jsem využil main konverzí.  

Main a Micro konverze se nachází ve složce results a jejich popis je následující:
```
========================= MAIN CONVERSIONS =========================
       PageID  VisitID     PageName  ... Length_seconds  Length_pagecount      Type
27       3065     1189      CATALOG  ...          540.0               7.0       NaN
49       3065     1200      CATALOG  ...           60.0               2.0  Fulltext
67       3065     1207      CATALOG  ...          120.0               4.0  Partners
111      3065     1226      CATALOG  ...           60.0               2.0       NaN
175      3118     1259  APPLICATION  ...          300.0               7.0  Fulltext
...       ...      ...          ...  ...            ...               ...       ...
38038    3065    17095      CATALOG  ...         1860.0               7.0  Fulltext
38053    3065    17101      CATALOG  ...           60.0               3.0  Partners
38090    3065    17113      CATALOG  ...         1320.0              13.0  Partners
38100    3065    17116      CATALOG  ...          480.0               9.0  Fulltext
38398    3065    17243      CATALOG  ...          240.0               5.0       NaN

[497 rows x 18 columns]
========================= MICRO CONVERSIONS ========================
       PageID  VisitID   PageName  ... Length_seconds  Length_pagecount       Type
37       3066     1192   DISCOUNT  ...          300.0               8.0  Catalogue
38       3066     1192   DISCOUNT  ...          300.0               8.0  Catalogue
56       3075     1204   WHOWEARE  ...          120.0               6.0   Fulltext
92       3097     1216  HOWTOJOIN  ...         1200.0              11.0   Fulltext
94       3099     1216   DISCOUNT  ...         1200.0              11.0   Fulltext
...       ...      ...        ...  ...            ...               ...        ...
37688    3075    16943   WHOWEARE  ...          420.0               6.0        NaN
37821    3066    17007   DISCOUNT  ...          780.0               9.0    OwnWebs
37843    3066    17014   DISCOUNT  ...           60.0               2.0        NaN
38217    3075    17162   WHOWEARE  ...           60.0               4.0  Catalogue
38397    3066    17243   DISCOUNT  ...          240.0               5.0        NaN

[442 rows x 18 columns]
```

## Apriori

K nalezení asociačních pravidel jsem použil [algoritmus ze cvičení 5](https://courses.fit.cvut.cz/NI-DDW/tutorials/05/index.html#_apriori-algorithm-implementation). 
Významých asociací nebylo mnoho a tak bylo potřeba si hrát s parametrem `support`. Nakonec jsem došel k hodnotě `0.05`. 

Výsledky jsem seřadil podle hodnot podpor:
```
============================== APRIORI =============================
frozenset({'CATALOG'}) - 1.123404255319149
frozenset({'TravelAgency'}) - 1.0659574468085107
frozenset({'TravelAgency', 'CATALOG'}) - 0.7021276595744681
frozenset({'lastminute'}) - 0.23829787234042554
frozenset({'lastminute', 'CATALOG'}) - 0.18936170212765957
frozenset({'hiking'}) - 0.17872340425531916
frozenset({'tours with tents'}) - 0.17872340425531916
frozenset({'tours and holiday comes into hotel'}) - 0.17659574468085107
frozenset({'sightseeing tours'}) - 0.14893617021276595
frozenset({'light hiking'}) - 0.14680851063829786
frozenset({'hiking', 'CATALOG'}) - 0.14680851063829786
frozenset({'tours with tents', 'CATALOG'}) - 0.14468085106382977
frozenset({'WHOWEARE'}) - 0.1425531914893617
frozenset({'Far tours'}) - 0.13617021276595745
frozenset({'lastminute', 'TravelAgency'}) - 0.13404255319148936
frozenset({'lastminute', 'TravelAgency', 'CATALOG'}) - 0.13191489361702127
frozenset({'CATALOG', 'sightseeing tours'}) - 0.125531914893617
frozenset({'TravelAgency', 'hiking'}) - 0.11914893617021277
frozenset({'TravelAgency', 'hiking', 'CATALOG'}) - 0.11914893617021277
frozenset({'light hiking', 'CATALOG'}) - 0.10851063829787234
frozenset({'tours and holiday comes into hotel', 'CATALOG'}) - 0.10851063829787234
frozenset({'Far tours', 'CATALOG'}) - 0.09787234042553192
frozenset({'Corsica'}) - 0.09361702127659574
frozenset({'cycling abroad'}) - 0.09148936170212765
frozenset({'CATALOG', 'WHOWEARE'}) - 0.09148936170212765
frozenset({'tours with tents', 'TravelAgency'}) - 0.08936170212765958
frozenset({'tours with tents', 'TravelAgency', 'CATALOG'}) - 0.08936170212765958
frozenset({'Alps tourism'}) - 0.08723404255319149
frozenset({'TravelAgency', 'sightseeing tours'}) - 0.08723404255319149
frozenset({'CATALOG', 'TravelAgency', 'sightseeing tours'}) - 0.0851063829787234
frozenset({'DISCOUNT'}) - 0.08297872340425531
frozenset({'Aeolian Islands'}) - 0.08085106382978724
frozenset({'expedition'}) - 0.08085106382978724
frozenset({'mountain expedition'}) - 0.07872340425531915
frozenset({'TravelAgency', 'WHOWEARE'}) - 0.07872340425531915
frozenset({'TravelAgency', 'tours and holiday comes into hotel'}) - 0.07872340425531915
frozenset({'TravelAgency', 'tours and holiday comes into hotel', 'CATALOG'}) - 0.07659574468085106
frozenset({'CATALOG', 'TravelAgency', 'WHOWEARE'}) - 0.07446808510638298
frozenset({'Corsica (stay with excursions) holidays 05'}) - 0.07234042553191489
frozenset({'CATALOG', 'Corsica'}) - 0.07021276595744681
frozenset({'Aeolian Islands', 'CATALOG'}) - 0.07021276595744681
frozenset({'TravelAgency', 'light hiking'}) - 0.06808510638297872
frozenset({'cycling abroad', 'CATALOG'}) - 0.06808510638297872
frozenset({'TravelAgency', 'light hiking', 'CATALOG'}) - 0.06808510638297872
frozenset({'expedition', 'CATALOG'}) - 0.06595744680851064
frozenset({'TravelAgency', 'Far tours'}) - 0.06382978723404255
frozenset({'hotelbuses'}) - 0.06170212765957447
frozenset({'stays'}) - 0.06170212765957447
frozenset({'Bulgaria'}) - 0.06170212765957447
frozenset({'stays with trips'}) - 0.06170212765957447
frozenset({'DISCOUNT', 'CATALOG'}) - 0.06170212765957447
frozenset({'TravelAgency', 'Far tours', 'CATALOG'}) - 0.06170212765957447
frozenset({'Corsica (stay with excursions) holidays 05', 'CATALOG'}) - 0.0574468085106383
frozenset({'stays with trips', 'CATALOG'}) - 0.05531914893617021
frozenset({'Alps tourism', 'CATALOG'}) - 0.05531914893617021
frozenset({'Bulgaria', 'CATALOG'}) - 0.05106382978723404
frozenset({'TravelAgency', 'Corsica'}) - 0.05106382978723404
frozenset({'CATALOG', 'stays'}) - 0.05106382978723404
```
Z výsledků je vidět, že nejnavštěvovanější stránky jsou typu `CATALOG` a s nazvem `TravelAgency`. Dálší významná čísla vidíme u `lastminute`. Vysoká podpora u `CATALOG` je samozřejmně důsledkem využíti main konverzí.

## Závěr

Hlavním problémem v tomto úkolu byla interpretace dat, převážně spojit je tak aby to dávalo smysl a vyvodit z toho závěry. Velkým pomocníkem byly pandas dataframy, které práci s daty ulehčili.

Program by se dal rozšířit o implikace s pradvěpodobnostmi a tyto výsledky potom interpretovat. Dovedu si představit, že výsledkem této analýzy, by mohlo být například sledování pohybů uživatelů po stránce a dynamické nabízení obsahu. Pokud někdo hledá lastminute tak je možná relevantní mu nabídnout TravelAgency atp.
