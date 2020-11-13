import re
import sys
import time
import requests

from bs4 import BeautifulSoup

MATHCHURL = 'lyrists_lyrics'
BASEURL = 'https://svr0.utamap.com/zi_list.php?sel_cat=0&sel_genre=05&page='
ADDRESS = 'https://svr0.utamap.com/'
PAGENUM = 100
TARGET = 5
TIME = 5
TAB = '\t'
BR = '<br>'


def write_lyric(title, lyric, fname):
    with open(fname, 'a') as fp:
        fp.write(title+TAB)
        for line in lyric:
            fp.write(line+TAB)
        fp.write('\n')

def remove_noise(lyric):
    lyric = re.sub(r"<(\"[^\"]*\"|'[^']*'|[^'\">])*>", "\n", lyric)
    words = lyric.split('\n')
    removed_lyric = []
    for word in words:
        if word != '\r' and word != '':
            removed_lyric.append(word)
    removed_lyric.pop()
    return removed_lyric


def get_lyric_one_html(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    lyric = ''
    for i, line in enumerate(soup.find_all('td', align='left')):
        try:
            target_contents = soup.find_all('td', align='left')[TARGET]
            target_contents = target_contents.contents
        except:
            target_contents = soup.find_all('td', align='left')[TARGET-1]
            target_contents = target_contents.contents
    for content in target_contents:
        lyric += str(content)
    lyric = remove_noise(lyric)
    return lyric


def get_urls(fname):
    for i in range(PAGENUM):
        res = requests.get(BASEURL + str(i))
        soup = BeautifulSoup(res.text, 'html.parser')
        for line in soup.find_all('a', href=re.compile(MATHCHURL)):
            time.sleep(TIME)
            title = line.contents[0]
            link = ADDRESS + line.attrs['href']
            lyric = get_lyric_one_html(link)
            write_lyric(title, lyric, fname)
            print(title, lyric)
        time.sleep(TIME)
    return 0


def main():
    urls = get_urls(sys.argv[1])


if __name__ == '__main__':
    main()
