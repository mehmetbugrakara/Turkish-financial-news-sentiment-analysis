import warnings

from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import time
from datetime import datetime, timedelta
import sentiment_analysis
warnings.filterwarnings('ignore')
# Scraping news articles from bloomberght.com

icerik = []
news_url_list = []
url_listesi = [
    "https://www.bloomberght.com/tum-ekonomi-haberleri",
    "https://www.bloomberght.com/tum-tarim-haberleri",
    "https://www.bloomberght.com/tum-finansal-teknoloji-haberleri",
    "https://www.bloomberght.com/tum-ekonomik-veriler-ve-gundem-haberleri",
    "https://www.bloomberght.com/tum-enerji-haberleri"
]

# Iterate through pages of each category and scrape news URLs
for sayi in tqdm(range(200)):
    try:
        for liste in url_listesi:
            time.sleep(10)
            url = urlopen(liste + "/" + str(sayi)).read()
            soup = BeautifulSoup(url, 'html.parser')
            container = soup.find_all('div', class_='widget-news-list type1')
            news_order = len(container[0].find_all('li'))

            for i in range(news_order):
                news_url = container[0].find_all('li')[i].find('a')['href']
                new_news = 'https://www.bloomberght.com' + news_url
                news_url_list.append(new_news)
    except:
        continue

# Create DataFrame to store news URLs
news_df = pd.DataFrame(news_url_list, columns=['url'])

# Iterate through each news URL and scrape content, title, and date
for links in tqdm(news_df['url']):
    try:
        time.sleep(10)
        url1 = urlopen(links).read()
        soup1 = BeautifulSoup(url1, 'html.parser')
        container1 = soup1.find_all('div', class_='news-item')
        news_date = container1[0].find('span', class_='date').text.split('\n')[1]
        news_title = container1[0].find('h1', class_='title').text.split('\n')[1]
        news_text = container1[0].find('article', class_='content').text
        icerik.append([news_date, news_title, news_text])
    except:
        continue

# Create DataFrame to store news content, title, and date
df = pd.DataFrame(icerik, columns=['datetime', 'title', 'content'])

# Convert the date format and filter news from the last week
format_str = '%d-%m-%Y'
date_time = []
for i in range(len(df)):
    dates = df.loc[i, 'datetime'].replace(" Aralık ", "-12-").replace(" Kasım ", "-11-").replace(" Ekim ", "-10-") \
        .replace(" Eylül ", "-9-").replace(" Ağustos ", "-8-").replace(" Temmuz ", "-7-").replace(" Haziran ", "-6-") \
        .replace(" Mayıs ", "-5-").replace(" Nisan ", "-4-").replace(" Mart ", "-3-").replace(" Şubat ", "-2-") \
        .replace(" Ocak ", "-1-")
    datesr = dates.split(" ")[0]
    date_time.append(datetime.strptime(datesr, format_str))

date_time = pd.DataFrame(date_time, columns=['date_times'])
now = datetime.now()
one_week = now - timedelta(weeks=4)
df = pd.concat([df, date_time], axis=1)
df.drop(columns=['datetime'])
df = df.loc[df['date_times'] > one_week]


data_dict = df.to_dict("records")

# Convert content to lowercase, perform sentiment analysis, and store the results in MongoDB
df['content'] = df['content'].str.lower()
df = df.dropna()
model = sentiment_analysis.Model('savasy/bert-base-turkish-sentiment-cased')
scored_df = model.sentiment_analysis(df, 'content', 'date_times')
scored_dict = scored_df.to_dict("records")
