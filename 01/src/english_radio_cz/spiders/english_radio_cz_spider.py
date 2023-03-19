import scrapy


class EnglishRadioCzSpiderSpider(scrapy.Spider):
    name = "english_radio_cz_spider"
    allowed_domains = ["english.radio.cz"]
    start_urls = ["https://english.radio.cz/science?page=0"]
    base_url = "https://english.radio.cz"

    def parse(self, response):
        articles = response.css('ul.b-004__list li.b-004__list-item')
        for article in articles:
            link = self.base_url + article.css('a::attr(href)').get()
            yield scrapy.Request(url=link, callback=self.parse_article)

        next_page_link = response.css('li.pager__item--next a::attr(href)').get()
        next_page_url = self.base_url + next_page_link
        if next_page_link:
            yield scrapy.Request(url=next_page_url, callback=self.parse)


    def parse_article(self, article):
        header = self.extract_header(article)
        body = self.extract_body(article)
        author = self.extract_author(article)
        publish_date = self.extract_publish_date(article)
        yield {
            "header": header,
            "body": body,
            "author": author,
            "publish_date": publish_date,
        }

    def extract_header(self, article):
        return article.css('h1.article-type ::text').get()

    def extract_body(self, article):
        return article.css('div.content-1-3 p::text').extract()

    def extract_author(self, article):
        return article.css('div.node-block--authors a::text').get()

    def extract_publish_date(self, article):
        return article.css('div.node-block__block--date span::text').get().strip()
