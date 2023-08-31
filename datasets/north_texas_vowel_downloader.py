#!/usr/bin/env python3
#
# This script download the North Texas Vowel dataset from its website.
# For some reason, they do not provide a zip archive, so we do it ourselves.
#

from html.parser import HTMLParser
import urllib.request
import os, sys

DATASET_URL = "http://www.utdallas.edu/~assmann/KIDVOW1/North_Texas_vowel_database.html"
FOLDER_NAME = 'ntv_dataset'


# Simple url extractor from
# https://stackoverflow.com/questions/6883049/regex-to-extract-urls-from-href-attribute-in-html-with-python
class UrlParser(HTMLParser):
    def __init__(self, output_list=None):
        HTMLParser.__init__(self)
        if output_list is None:
            self.output_list = []
        else:
            self.output_list = output_list

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.output_list.append(dict(attrs).get('href'))


# Create parser
p = UrlParser()


# Get webpage content
def query_file(url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
            'AppleWebKit/537.11 (KHTML, like Gecko) '
            'Chrome/23.0.1271.64 Safari/537.11',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'}
    req = urllib.request.Request(url=url, headers=headers)
    contents = urllib.request.urlopen(req).read()
    return contents


web_page_contents = query_file(DATASET_URL).decode('utf-8', 'ignore')

# Extract urls
p.feed(web_page_contents)
files = [s for s in p.output_list if s.endswith('.wav') or s.endswith('.WAV')]

os.makedirs(FOLDER_NAME, exist_ok=True)
for file in files:
    print("Request %s" % file)
    data = query_file(file)
    filename = os.path.basename(urllib.parse.urlparse(file).path)
    open(f"{FOLDER_NAME}/{filename}", "wb").write(data)
