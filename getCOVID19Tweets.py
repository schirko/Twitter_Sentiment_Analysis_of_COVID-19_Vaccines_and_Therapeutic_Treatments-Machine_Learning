import itertools
from nltk import collections
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import tweepy as tw
import pandas as pd
import csv
import re
from datetime import datetime
from uuid import uuid4


class getTweets:

    def __init__(self):
        self.tweet = []
        self.tweet_words = []
        self.num_tweets = 0
        self.num_vaccine_tweets = 303  # self.num_tweets / 2
        self.num_therapeutics_tweets = 120  # self.num_tweets / 4
        self.num_treatment_tweets = 0
        self.num_tweets_counter = 0
        self.num_tweets_vaccine_count = 0
        self.num_tweets_therapeutics_count = 0
        self.vaccine_tweets_df = []
        self.vaccine_positive_tweets_df = []
        self.vaccine_neutral_tweets_df = []
        self.vaccine_negative_tweets_df = []
        self.therapeutics_tweets_df = []
        self.therapeutics_negative_tweets_df = []
        self.therapeutics_positive_tweets_df = []
        self.therapeutics_neutral_tweets_df = []

    def process_tweets(self):

        # Create CSV file with all tweets for all treatments
        pd.set_option('display.max_columns', None)
        csv_file_all = open(all_tweets_filename, "w", newline="")  # newline prevents blank lines in CSV
        csv_tweets_all = csv.writer(csv_file_all)
        csv_tweets_all.writerow(csv_headers)

        print("\n" + "Treatment: Covid-19 Vaccine")
        self.num_tweets_vaccine_count += 1
        self.num_tweets = self.num_vaccine_tweets

        for self.tweet in tw.Cursor(api.search, q="#covid-19 vaccine" + search_filter,
                                    lang="en",
                                    since=since_date
                                    ).items(self.num_tweets):
            # Pre-processing Tweets
            self.clean_tweet()  # Remove unwanted characters, URLs, usernames, and words
            self.get_sentiments("covid-19 vaccine")  # Get actual Polarity & Subjectivity, and binary Sentiment values
            self.tokenize_stopwords()  # Tokenize the tweet and remove stopwords

            # Analyzing and InsertingTweets
            words_in_tweet = [tweet.split() for tweet in self.tweet_words]  # split tweets into words
            all_words = list(itertools.chain(*words_in_tweet))  # List of all words across tweets
            count_words = collections.Counter(all_words)  # Create counter
            split_tweets = pd.DataFrame(count_words.most_common(5), columns=['words', 'count'])
            #print(split_tweets.head(n=25))  # Display frequent words
            self.insert_tweets(csv_tweets_all)  # Insert tweets into CSV appropriate files

        # Vaccine Sentiments
        vaccine_df = pd.read_csv(all_tweets_filename, sep=',')
        print("vaccine_df", vaccine_df)
        self.vaccine_tweets_df = vaccine_df.loc[vaccine_df['treatment'] == 'covid-19 vaccine']
        self.vaccine_positive_tweets_df = self.vaccine_tweets_df.loc[self.vaccine_tweets_df['sentiment'] == 'Positive']
        self.vaccine_neutral_tweets_df = self.vaccine_tweets_df.loc[self.vaccine_tweets_df['sentiment'] == 'Neutral']
        self.vaccine_negative_tweets_df = self.vaccine_tweets_df.loc[self.vaccine_tweets_df['sentiment'] == 'Negative']
        print("Total Vaccine: ", len(self.vaccine_tweets_df['treatment']))
        print("Total Vaccine Positive: ", len(self.vaccine_positive_tweets_df))
        print("Total Vaccine Neutral: ", len(self.vaccine_neutral_tweets_df))
        print("Total Vaccine Negative: ", len(self.vaccine_negative_tweets_df))


        # Loop through Therapeutic treatments
        for treatment in treatments:

            print("\n" + "Therapeutics: " + treatment)

            # Get more tweets if the counter levels have not been met
            if self.num_tweets_counter < self.num_therapeutics_tweets:

                self.num_tweets = self.num_therapeutics_tweets - self.num_tweets_counter
                for self.tweet in tw.Cursor(api.search, q="#" + treatment + search_filter,
                                            lang="en",
                                            since=since_date
                                            ).items(self.num_tweets):

                    # Pre-processing Tweets
                    self.clean_tweet()  # Remove unwanted characters, URLs, usernames, and words
                    self.get_sentiments(treatment)  # Get actual Polarity & Subjectivity, and binary Sentiment values
                    self.tokenize_stopwords()  # Tokenize the tweet and remove stopwords
                    self.insert_tweets(csv_tweets_all)  # Insert tweets into CSV appropriate files
                    self.num_tweets_counter += 1

        # Therapeutic sentiments
        therapeutics_tweets_df = pd.read_csv(all_tweets_filename, sep=',')
        all_therapeutics_tweets_df = therapeutics_tweets_df.loc[
            therapeutics_tweets_df['treatment'] != 'covid-19 vaccine']
        self.therapeutics_positive_tweets_df = all_therapeutics_tweets_df.loc[all_therapeutics_tweets_df['Positive'] == 'Positive']
        self.therapeutics_neutral_tweets_df = all_therapeutics_tweets_df.loc[all_therapeutics_tweets_df['Neutral'] == 'Neutral']
        self.therapeutics_negative_tweets_df = all_therapeutics_tweets_df.loc[all_therapeutics_tweets_df['Negative'] == 'Negative']
        print("Total Therapeutics: ", len(all_therapeutics_tweets_df['treatment']))
        print("Total Therapeutics Positive: ", len(self.therapeutics_positive_tweets_df))
        print("Total Therapeutics Neutral: ", len(self.therapeutics_neutral_tweets_df))
        print("Total Therapeutics Negative: ", len(self.therapeutics_negative_tweets_df))

    def clean_tweet(self):
        # Clean up and normalize tweets
        tweet_text = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", self.tweet.text)  # Remove URLs
        tweet_text = re.sub('@[^\s]+', '', tweet_text)  # Remove twitter usernames
        tweet_text = re.sub(r"[^a-zA-Z0-9]+", ' ', tweet_text)  # Remove special characters
        tweet_text = re.sub(r"\b[a-zA-Z]\b", "", tweet_text)  # Remove single character words
        tweet_text = tweet_text.lower()  # Casing the tweets
        tweet_text = tweet_text.replace("covid 19", 'covid19')  # Make 'COVID19' easier to search, dashes already gone
        self.tweet.text = tweet_text
        return self.tweet.text

    def tokenize_stopwords(self):
        # Tokenize and Remove Stop Words
        tweet_tokens = word_tokenize(self.tweet.text)
        self.tweet_words = [word for word in tweet_tokens if not word in stopwords.words()]
        return self.tweet_words

    def get_sentiments(self, treatment):
        # Process tweets for their polarity and subjectivity values
        self.tweet.polarity = TextBlob(self.tweet.text).sentiment.polarity  # Actual sentiment
        self.tweet.subjectivity = TextBlob(self.tweet.text).sentiment.subjectivity  # Actual subjectivity
        if self.tweet.polarity < 0:  # Process binary sentiment values
            self.tweet.sentiment = 'Negative'
        elif self.tweet.polarity == 0:
            self.tweet.sentiment = 'Neutral'
        else:
            self.tweet.sentiment = 'Positive'
        self.tweet.treatment = treatment
        return self.tweet

    def insert_tweets(self, csv_treatment_tweets_all):  #
        # Insert tweet and sentiment data into CSV
        tweet_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())  # Generate unique id for tweets
        tweet_words = ' '.join(self.tweet_words)  # Convert list of words back to a string
        if tweet_words != "":  # Insert of tweet words field is not empty
            csv_treatment_tweets_all.writerow([self.tweet.created_at,
                                               self.tweet.treatment, self.tweet.sentiment, self.tweet.polarity,
                                               self.tweet.subjectivity, self.tweet.text, tweet_words])
        return self.tweet, csv_treatment_tweets_all


if __name__ == "__main__":
    # Twitter API credentials
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""

    # Create Twitter authentication handler
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Define search variables
    treatments = ["regeneron", "hydroxychloroquine", "remdesivir"]
    csv_headers = ["created_at", "treatment", "sentiment", "polarity", "subjectivity", "text", "tweet_words"]
    search_filter = " -filter:retweets"
    all_tweets_filename = 'data/TweetsBySearchTerm.csv'
    since_date = '2020-01-01'

    # Scopes
    gt = getTweets()
    gt.process_tweets()
