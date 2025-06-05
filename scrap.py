import os
import re
import requests
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from agent.constants import DATA_FOLDER

os.makedirs(DATA_FOLDER, exist_ok=True)
BASE_URL = "https://sede.agenciatributaria.gob.es"

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

    text = ""
    for x in header.next_siblings:
        if type(x) is Tag and x.name != 'div':
            item_text = markdownify(str(x))
            if item_text:
                text += " " + x.text + "\n"

    return {
        "title": title,
        "text": text
    }


def print_doc(prev_text, item, doc_num):
    page_num = 0
    titulo = item.get('title', '')
    if 'items' in item:
        for i in item['items']:
            if prev_text:
                print_doc(f"{prev_text} - {titulo}", i, doc_num)
            else:
                print_doc(titulo, i, doc_num)
    else:
        if prev_text:
            doc_title = f"{prev_text} - {titulo}"
        else:
            doc_title = titulo
        with open(f"{DATA_FOLDER}/doc_{doc_num}_page_{page_num}.txt", "w", encoding="utf8") as f:
            text = re.sub(r'\n\n+', r'\n', item['text'])
            f.write(f"{doc_title}\n\n{text}")
        page_num += 1


if __name__ == '__main__':
    for doc_num, url in enumerate([
        "/Sede/Ayuda/24Manual/100.html",
        "/Sede/Ayuda/24Manual/100/deducciones-autonomicas.html"
    ]):
        data = parse_web(url)
        for i in data['items'][1:]:
            print_doc("", i, doc_num)
