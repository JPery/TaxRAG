import os
import re
import sys
import requests
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from agent.constants import DATA_FOLDER

BASE_URL = "https://sede.agenciatributaria.gob.es"

if len(sys.argv) > 1:
    DATA_FOLDER = sys.argv[1]

os.makedirs(DATA_FOLDER, exist_ok=True)
checked_urls = set()


def parse_web(url):
    if not url.startswith("/") or url in checked_urls:
        return
    html_doc = requests.get(BASE_URL + url).content.decode("utf8")
    checked_urls.add(url)
    soup = BeautifulSoup(html_doc, 'html.parser')

    main = soup.find(id="acc-main")
    header = main.find("h1")
    title = header.text
    elements = main.find("ol")
    if elements is not None:
        links = elements.find_all("a")
        if links:
            data = []
            for link in elements.find_all("a"):
                href = link.attrs['href']
                if href is not None:
                    new_data = parse_web(href)
                    if new_data:
                        data.append(new_data)
            return {
                'title': title,
                'items': data
            }

    new = BeautifulSoup('')
    for x in header.next_siblings:
        if type(x) is Tag and x.name != 'div':
            new.append(x)
    text = markdownify(str(new), heading_style="ATX", bullets="-")
    if title and text:
        return {
            "title": title,
            "text": text
        }


def print_doc(prev_text, item, doc_num):
    global page_num
    title = item.get('title', '')
    if prev_text:
        doc_title = f"{prev_text} - {title}"
    else:
        doc_title = title
    if 'items' in item:
        for i in item['items']:
            print_doc(doc_title, i, doc_num)
    else:
        with open(f"{DATA_FOLDER}/doc_{doc_num}_page_{page_num}.txt", "w", encoding="utf8") as f:
            text = re.sub(r'\n\n+', r'\n', item['text'])
            f.write(f"# {doc_title}\n{text}")
        page_num += 1


if __name__ == '__main__':
    for doc_num, (url, prev_text) in enumerate([
        ("/Sede/Ayuda/24Manual/100.html", ""),
        ("/Sede/Ayuda/24Manual/100/deducciones-autonomicas.html", "Deducciones autonómicas")
    ]):
        page_num = 0
        data = parse_web(url)
        for i in data['items']:
            print_doc(prev_text, i, doc_num)
