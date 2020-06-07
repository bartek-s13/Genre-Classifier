import logging

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
import spacy
from spacy_langdetect import LanguageDetector
import os
import sys
import lyricsgenius
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import operator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas
import numpy as np


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
    for f in html.find_all('div', class_="lyrics"):
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


def scrap_song_az_lyrics(url: str):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    for f in html.find_all('div', class_='col-xs-12 col-lg-8 text-center'):
        if f is not None:
            return f.find_all('div')[5].text
    return None


def scrapping_lyrics(title: str, artist: str) -> str:
    return scrap_song_az_lyrics(
        f"https://www.azlyrics.com/lyrics/{''.join(artist.lower().split())}/{''.join(title.lower().split())}.html")


def get_songs(filename: str):
    data = pd.read_csv("data/titles/" + filename + ".csv")
    return zip(data.artist, data.name)


def create_pipeline(X, y, ngram=(1, 1), df=0.4):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=300, ngram_range=ngram, max_df=df)),
        ('scaler', StandardScaler(with_mean=False)),
        # ('svr', svm.SVR()),
        ('svc', svm.SVC(kernel="linear"))
    ])
    pipeline.fit(X, y)
    return pipeline


def normalize_lyric(lyric: str):
    return [i for i in word_tokenize(lyric) if 'A' <= i <= 'z']   # Usuwanie interpunkcji i tokenizacja

    # TODO sprawdzanie języka na szybko, w get_lyrics jak chciałem dodać warunek w if, to nie zawsze działało, bo się zapętlało
    # if lyric is not None and check_if_english(lyric):
    #     lyric = re.sub(r'\[.+?\]', r'', lyric)
    #     return re.sub(r'\n', r' ', lyric).strip()
    return None


def genre_indicators(feature_names, feature_importances, id_to_genre):  # TODO: do naprawienia (svc nie jest liniowy, nie można pobrac coef)
    for i, genre in enumerate(feature_importances):
        scored_features = list(zip(feature_names, genre.data))
        scored_features = sorted(scored_features, key=lambda x: x[1], reverse=True)
        print("W rozpoznaniu gatunku {lang} najważniejsze cechy to:".format(lang=id_to_genre[i]))
        for feature, score in scored_features[:5]:
            print("\t'{feature}': {score}".format(feature=feature, score=score))


def get_lyrics_from_file(genre: str, file: str) -> str:
    try:
        with open(f"data/lyrics/{genre}/{file}", 'r') as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    map_chars = {b'\x92': "'", b'\xe9': chr(ord(b'\xe9'))}
    with open(f"data/lyrics/{genre}/{file}", 'br') as f:
        sb = f.read()
    ret = ''
    for i in [i.to_bytes(1, sys.byteorder) for i in sb]:
        try:
            ret += i.decode('utf8')
        except UnicodeDecodeError:
            if i in map_chars:
                ret += map_chars[i]
    return ret


def check_if_english(lyrics: str) -> bool:
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    doc = nlp(lyrics)
    return doc._.language['language'] == 'en'


def get_top_artists(filename):
    data = pd.read_csv("data/titles/" + filename + ".csv")
    count = data.groupby("artist")["name"].count().reset_index(name='count').sort_values(['count'], ascending=False)
    return count.artist.values


def lyrics_downloader(filename, songs_per_artist=6, songs_num=3000):
    genius = lyricsgenius.Genius("9y4ky-47cpjAgFmIljdSVhT306jZsIVF4NfQ43eTQ-cjd3yXrzTAdxjkttCUV0SN")
    genius.verbose = False
    genius.remove_section_headers = True

    already_downloaded = 0
    artists = get_top_artists(filename)
    os.chdir('./data/lyrics/' + filename)
    for artist in artists:
        try:
            art = genius.search_artist(artist, max_songs=songs_per_artist)
            for song in art.songs:
                try:
                    title = '_'.join(song.title.split())
                    song.to_text(filename=title + ".txt")
                    logging.info(f"{song.title} saved")
                    already_downloaded += 1
                    logging.info(f"{already_downloaded} out of {songs_num} downloaded")
                    if already_downloaded == songs_num:
                        break
                except:
                    logging.error(f"{song.title} not saved")
            if already_downloaded == songs_num:
                break
        except:
            logging.ERROR("Artist: {} was not found".format(artist.name))


if __name__ == "__main__":
    genre = ["pop", "rock_metal", "rap_hip_hop"]
    data = []
    for i, gen in enumerate(genre):
        mypath = f"data/lyrics/{gen}"
        for j in [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]:
            data.append([get_lyrics_from_file(gen, j), i])
    data = np.array(data)
    data = data[data[:, 0] != '']
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1].astype('int'), test_size=0.30)

    n_grams = (1, 3)
    max_df = 0.3
    pip = create_pipeline(X_train, y_train, n_grams, max_df)
    print(f"\nngrams: {n_grams} \t max_df: {max_df} \t score: {pip.score(X_test, y_test)}\n")

    genre_indicators(pip.named_steps['tfidf'].get_feature_names(), pip.named_steps['svc'].coef_, genre)


    # get_songs("rap_hip_hop")
    # tekstowo_artist("A$AP Rocky & Tom Morello")
    # for a in get_songs("rap_hip_hop"):
    #     tekstowo_artist(a)
    # get_lyrics("Ms. Jackson", "OutKast")
    # songs = get_songs("rap_hip_hop")
    # for z in songs:
    #     print(str(z[0]) + '\t' + str(z[1]))
    # lyric = normalize_lyric(get_lyrics(str(z[1]), str(z[0])))
    # if lyric is not None:
    # print(lyric)
    # break
    # lyric = scrapping_lyrics("Since u been gone", "Kelly Clarkson")
    # if lyric:
    #     print(lyric)
    # break
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # import sys
    #
    # ch = logging.StreamHandler(sys.stdout)
    # ch.setLevel(logging.INFO)
    # logger.addHandler(ch)
    # # for filename in os.listdir("data/titles/"):
    # # lyrics_downloader(filename)
    # lyrics_downloader('rock_metal')
