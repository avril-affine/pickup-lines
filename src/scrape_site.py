import requests
from bs4 import BeautifulSoup


def get_rows(url):
    print url
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    table = soup.find('table')
    rows = table.findAll('td', {'class': 'column-1'})
    return rows[:-1]


def get_links():
    url = 'https://pickupline.net/'

    rows = get_rows(url)
    links = [row.find('a').get('href') for row in rows]
    links = [link for link in links if link.startswith(url)]

    return links


def get_lines(links):
    all_lines = []
    for url in links:
        if 'spring-break' in url:
            continue
        rows = get_rows(url)
        lines = [row.text for row in rows]
        all_lines.extend(lines)

    return all_lines


def main():
    links = get_links()
    lines = get_lines(links)
    with open('data2.txt', 'w') as f:
        for line in lines:
            # f.write(repr(line.encode('utf-8')))
            # f.write('\n')
            f.write(line.encode('utf-8') + '\n')


if __name__ == '__main__':
    main()
