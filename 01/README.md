# HW1 - Data Acquisition - Web Crawler/Scraper

https://courses.fit.cvut.cz/NI-DDW/hw/01/index.html

___

Web pro robota jsem vybral https://english.radio.cz/science.

Podle https://english.radio.cz/robots.txt jsem vyhodnotil, že crawlingu, krom `Crawl-delay: 10`, nic nebrání. Toto bylo zohledněno a nastaveno v souboru [src/english_radio_cz/settings.py](./src/english_radio_cz/settings.py).  
Mezi https://english.radio.cz/sitemap.xml jsem nenalezl pro sekci **Science and technology** nic zajimavého.

Skript podle parametru `DEPTH_LIMIT = 20` v souboru [src/english_radio_cz/settings.py](./src/english_radio_cz/settings.py) přečte 20 stránek z https://english.radio.cz/science. Na každé stránce otevře články a z nich vyextrahuje nadpis, textový obsah článku, autora a datum.

Scrapenutá data se nachází v [results/data.json](./results/data.json)

### Spuštění:
```
python src/main.py
```
