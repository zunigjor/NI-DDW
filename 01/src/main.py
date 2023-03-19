from scrapy import cmdline

cmdline.execute("scrapy crawl english_radio_cz_spider -o ../results/data.json".split())
