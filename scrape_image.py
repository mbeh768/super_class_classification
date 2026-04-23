from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "ligers/scraped"})
crawler.crawl(keyword="liger", max_num=50)