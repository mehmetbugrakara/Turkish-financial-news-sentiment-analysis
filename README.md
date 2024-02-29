## Turkish Financial News Sentiment Analyzer

Turkish Financial News Sentiment Analyzer is a Python-based project that collects and analyzes Turkish financial news articles from sources like BloombergHT. It performs web scraping to gather news URLs and their content, and then applies sentiment analysis and text classification using a BERT-based Turkish sentiment analysis model.

# Features

Collects URLs of Turkish financial news articles from BloombergHT.
Scrapes news content using web scraping tools like Beautiful Soup and urllib.
Utilizes a BERT-based Turkish sentiment analysis model (sentiment_analysis) for sentiment analysis.
Filters news articles based on date to focus on recent content.
Provides insights into sentiment trends in Turkish financial news.

# How to Use

Clone the repository to your local machine.
Install the required dependencies using "pip install -r requirements.txt".
Run the "bloomberght_scraping.py" script to collect news articles, perform sentiment analysis.
Explore the sentiment analysis results and trends using your preferred data analysis tools.

# Requirements
Python 3.x
Required Python packages listed in requirements.txt

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
Thanks to the developers of Beautiful Soup, urllib, and other libraries used in this project.
Special thanks to the creators of the BERT-based Turkish sentiment analysis model used for sentiment analysis.