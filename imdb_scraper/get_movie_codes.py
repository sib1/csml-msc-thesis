#!/usr/bin/env python3
import re
import csv
import json
import random
import urllib.request

from joblib import Parallel, delayed
from bs4 import BeautifulSoup
from lxml.html.clean import clean_html
import pickle



# Randomly select browser and referrer heads to avoid getting blocked
BROWSERS = ['Mozilla/5.0 (X11; U; Linux x86_64; en-US; rv:1.9.1.3) Gecko/20090913 Firefox/3.5.3',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en; rv:1.9.1.3) Gecko/20090824 Firefox/3.5.3 (.NET CLR 3.5.30729)',
            'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US; rv:1.9.1.3) Gecko/20090824 Firefox/3.5.3 (.NET CLR 3.5.30729)',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.1) Gecko/20090718 Firefox/3.5.1',
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.1 (KHTML, like Gecko) Chrome/4.0.219.6 Safari/532.1',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; InfoPath.2)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; SLCC1; .NET CLR 2.0.50727; .NET CLR 1.1.4322; .NET CLR 3.5.30729; .NET CLR 3.0.30729)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.2; Win64; x64; Trident/4.0)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; SV1; .NET CLR 2.0.50727; InfoPath.2)',
            'Mozilla/5.0 (Windows; U; MSIE 7.0; Windows NT 6.0; en-US)',
            'Mozilla/4.0 (compatible; MSIE 6.1; Windows XP)',
            'Opera/9.80 (Windows NT 5.2; U; ru) Presto/2.5.22 Version/10.51']

REFERRER = ['http://www.google.com/?q=',
            'http://www.usatoday.com/search/results?q=',
            'http://engadget.search.aol.com/search?q=',
            'https://www.bing.com/search?q=']
def download_html(url, timeout=10):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', random.choice(BROWSERS))
    req.add_header('Accept-Language', 'en-US,en;q=0.5')
    req.add_header('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.7')
    req.add_header('Cache-Control', 'no-cache')
    req.add_header('Referer', random.choice(REFERRER))
    resp = urllib.request.urlopen(req, timeout=timeout)
    html = resp.read()
    return html



def parse_html(html):
    out = {}
    soup = BeautifulSoup(html, "lxml")
    for script in soup(["script", "style"]):
        script.extract()

    reviews_soup = soup.findAll('div', {'class': 'text show-more__control'})
    reviews_soup = [r.text.strip() for r in reviews_soup]
    movie_title = soup.find('a', {'itemprop': 'url'}).text
    return movie_title, reviews_soup


def get_movie_codes(url, save_to_file=True):
    try:
        html = download_html(url)
        movie_title, reviews = parse_html(html)

        if save_to_file:
            filename = ' '.join(movie_title.split())  # replace all whitespace with single space
            filename = re.sub(r'[^\w\s]', '', filename)
            filename = filename.replace(' ', '_').lower().strip()
            with open(f'crawled_data/{filename}.json', 'w') as f:
                json.dump(reviews, f)
        else:
            return reviews

    except Exception:
        print(f'RAISED EXCEPTION FOR URL: {url}')

#file = download_html(url='https://www.imdb.com/search/title/?title_type=feature&release_date=2015-01-01,2019-12-31&ref_=adv_prv')

#file = download_html(url='https://www.imdb.com/search/title/?title_type=feature&release_date=2015-01-01,2019-12-31&start=51&ref_=adv_nxt')




codes = []

num_pages = 200

for j in range(num_pages):

	print(j)

	file = download_html(url='https://www.imdb.com/search/title/?title_type=feature&release_date=2012-01-01,2014-12-31&start=' +str(j*50 +1) + '&ref_=adv_nxt')
	soup = BeautifulSoup(file, "lxml")
	links = soup.find_all('h3',class_="lister-item-header")
	for i in range(50):
		line = links[i].contents[3]
		codes.append(str(line)[16:25])



with open('imdb_movie_codes_2', 'wb') as fp:
    pickle.dump(codes, fp)


"""
with open ('movie_codes', 'rb') as fp:
    codes = pickle.load(fp)


a = 'https://www.imdb.com/title/'
b = '/reviews?ref_=tt_ql_3'

crawler_input = [ a + c + b for c in codes]

print(len(crawler_input))




"""



