from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import pandas as pd
import time
from datetime import datetime, timedelta

class WebScraping:
    def __init__(self, html):
        self.links = html

    def scrap(self, soup, class_, class_name):
        """Scrapes the content of a specific class from the given HTML soup.

        Args:
        soup (BeautifulSoup): The parsed HTML soup.
        class_ (str): The HTML tag containing the class.
        class_name (str): The name of the class to be scraped.

        Returns:
        str: Scraped content.
        """
        scraped_content = soup.find(class_, {'class': class_name}).text
        return scraped_content

    def collection(self, news_, header_, date_time_, categor_, icerik, categories):
        """Collects scraped data into lists.

        Args:
        news_ (str): Scraped news content.
        header_ (str): Scraped news header.
        date_time_ (str): Scraped news date and time.
        categor_ (str): Category of the news.
        icerik (list): List to store content, date, and header.
        categories (list): List to store categories.

        Returns:
        tuple: Updated lists (icerik and categories).
        """
        news_ = news_.replace('\n', ' ')
        header_ = header_.replace('\n', ' ')
        icerik.append([header_, date_time_, news_])
        if str(news_) != '[]':
            categories.append([categor_])
        return icerik, categories

    def scraping(self, sub_cat, scrap, collection):
        """Scrapes data from a website.

        Args:
        sub_cat (str): Sub-category to scrape.
        scrap (function): Function to scrape data.
        collection (function): Function to collect scraped data.

        Returns:
        tuple: Scraped content, links, and sub-categories.
        """
        # HTML class list
        class_list = ['box only-text border-bottom', 'box image-right_text-left border-bottom responsive',
                      'box only-text d-block', 'box only-text border-bottom',
                      'box image-left_text-right border-bottom half-margin_bottom']
        icerik = []
        categories = []
        linkler = []
        sub_cats_name = []

        for sayi in tqdm(range(1)):
            time.sleep(2)
            html_text = requests.get(f"{self.links}{sub_cat}/{sayi}").text
            soup1 = BeautifulSoup(html_text, 'html.parser')
            for i in soup1.find_all('a', class_=class_list):
                links = i.get('href')
                check = links.split("/")[2]
                if check != 'www.dunyainsaat.com.tr' and check != 'www.dunyagida.com.tr' \
                        and check != 'www.makinamagazin.com.tr' and check != 'www.computerworld.com.tr':
                    url = requests.get(links).text
                    categor = links.split("/")[3]
                    linkler.append(links)
                    soup = BeautifulSoup(url, 'html.parser')
                    if soup.find('div', {'class': 'content-text'}) is not None:
                        news = scrap(soup, 'div', 'content-text')
                        header = scrap(soup, 'header', 'col-12')
                        date_time = scrap(soup, 'time', 'pubdate')
                        output_1 = collection(news, header, date_time, categor, icerik, categories)
                        try:
                            if str(news) != '[]':
                                sub_cats_name.append(sub_cat)
                        except:
                            print('kobiden')
                    else:
                        continue
                else:
                    continue
        return output_1, linkler, sub_cats_name

sektorler = WebScraping("https://www.dunya.com/sektorler")
advanced_news = WebScraping("https://www.dunya.com")
sub_cats = ['/otomotiv', '/emlak', '/tarim', '/turizm', '/sigortacilik', '/teknoloji', '/tekstil', '/madencilik',
            '/lojistik', '/gida', '/makine', '/iklimlendirme', '/enerji']
categ = ['/ekonomi', '/gundem', '/dunya']

def to_dataframe(total):
    """Converts scraped data into a DataFrame.

    Args:
    total (tuple): Scraped content, links, and sub-categories.

    Returns:
    DataFrame: DataFrame containing the scraped data.
    """
    haber_icerik = pd.DataFrame(total[0][0], columns=['Header', 'Date Time', 'Content'])
    categories_ = pd.DataFrame(total[0][1], columns=['Category'])
    linklerim = pd.DataFrame(total[1], columns=['links'])
    subs_cat = pd.DataFrame(total[2], columns=['Sub Cat'])
    alltogether = pd.concat([categories_, subs_cat, haber_icerik, linklerim], axis=1)
    alltogether.dropna(inplace=True)
    return alltogether

for sub_cat in sub_cats:
    cikis_sektorler = sektorler.scraping(sub_cat, sektorler.scrap, sektorler.collection)

for cats in categ:
    cikis_news = advanced_news.scraping(cats, advanced_news.scrap, advanced_news.collection)

sektorler_df = to_dataframe(cikis_sektorler)
news_df = to_dataframe(cikis_news)
dunya_data = pd.concat([sektorler_df, news_df])
dunya_data = dunya_data.reset_index()

def change_date_type(data):
    """Changes the date type in the DataFrame.

    Args:
    data (DataFrame): DataFrame containing date time information.

    Returns:
    DataFrame: DataFrame with date time column converted to datetime type.
    """
    format_str = '%d-%m-%Y'
    date_time = []
    for i in range(len(data)):
        dates = data['Date Time'].values[i].replace(" Aralık ", "-12-").replace(" Kasım ", "-11-") \
                    .replace(" Ekim ", "-10-").replace(" Eylül ", "-9-").replace(" Ağustos ", "-8-") \
                    .replace(" Temmuz ", "-7-").replace(" Haziran ", "-6-").replace(" Mayıs ", "-5-") \
                    .replace(" Nisan ", "-4-").replace(" Mart ", "-3-").replace(" Şubat ", "-2-") \
                    .replace(" Ocak ", "-1-")
        datesr = dates.split(" ")[0]
        date_time.append(datetime.strptime(datesr, format_str))

    date_time = pd.DataFrame(date_time, columns=['date_times'])
    all_datas = pd.concat([data, date_time], axis=1)

    return all_datas

all_data = change_date_type(news_df)

def take_spesific_times(data):
    """Filters the DataFrame to include only recent data.

    Args:
    data (DataFrame): DataFrame containing date time information.

    Returns:
    DataFrame: DataFrame with only recent data.
    """
    now = datetime.now()
    one_week = now - timedelta(weeks=1)
    data = data.loc[data['date_times'] > one_week]
    return data

all_data = take_spesific_times(all_data)
