import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
import spacy
from spacy_langdetect import LanguageDetector


def request_song_info(song_title, artist_name):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + '9y4ky-47cpjAgFmIljdSVhT306jZsIVF4NfQ43eTQ-cjd3yXrzTAdxjkttCUV0SN'}
    search_url = base_url + '/search'
    data = {'q': song_title + ' ' + artist_name}
    response = requests.get(search_url, data=data, headers=headers)

    return response


def scrap_song_url(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    # [h.extract() for h in html('script')]
    lyrics = ""
    for f in html.find_all('div', class_='lyrics'):
        if f is not None:
            lyrics = f.get_text()
    # f = html.find('div', class_='lyrics')
    # lyrics = f.get_text()

    return lyrics


def get_lyrics(title, artist):
    response = request_song_info(title, artist)
    json = response.json()
    remote_song_info = None

    for hit in json['response']['hits']:
        if artist.lower() in hit['result']['primary_artist']['name'].lower():
            remote_song_info = hit
            break

    if remote_song_info:
        song_url = remote_song_info['result']['url']
        lyrics = scrap_song_url(song_url)
        if lyrics != "":
            return lyrics
        else:
            return get_lyrics(title,
                              artist)  # wywołuje to tak, bo czasami z niewiadomych mi powodów nie wyszukuje niczego
    return None


def get_songs(filename: str):
    data = pd.read_csv("data/titles/" + filename + ".csv")
    return zip(data.artist, data.name)


def normalize_lyric(lyric: str):  # TODO: usuwanie stopword i interpunkcji, ale to po tokenizacji

    #TODO sprawdzanie języka na szybko, w get_lyrics jak chciałem dodać warunek w if, to nie zawsze działało, bo się zapętlało
    if lyric is not None and check_if_english(lyric):
        lyric = re.sub(r'\[.+?\]', r'', lyric)
        return re.sub(r'\n', r' ', lyric).strip()
    return None


def check_if_english(lyrics: str) -> bool:
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    doc = nlp(lyrics)
    return doc._.language['language'] == 'en'



if __name__ == "__main__":
    # get_songs("rap_hip_hop")
    # tekstowo_artist("A$AP Rocky & Tom Morello")
    # for a in get_songs("rap_hip_hop"):
    #     tekstowo_artist(a)
    # get_lyrics("Ms. Jackson", "OutKast")
    songs = get_songs("rap_hip_hop")
    for z in songs:
        print(str(z[0]) + '\t' + str(z[1]))
        lyric = normalize_lyric(get_lyrics(str(z[1]), str(z[0])))
        if lyric is not None:
            print(lyric)
        #break