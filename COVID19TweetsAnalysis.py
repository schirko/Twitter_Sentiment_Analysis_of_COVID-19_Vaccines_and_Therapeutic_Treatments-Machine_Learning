import itertools
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn import datasets, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
from nltk import collections
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pandas as pd


class getTweets:

    def __init__(self):
        self.tweets = []
        self.therapeutics_tweets_df = []
        self.vaccine_tweets_df = []
        self.vaccine_positive_tweets_df = []
        self.vaccine_neutral_tweets_df = []
        self.vaccine_negative_tweets_df = []
        self.therapeutics_positive_tweets_df = []
        self.therapeutics_neutral_tweets_df = []
        self.therapeutics_negative_tweets_df = []

    # Exploratory Data Analysis
    def process_tweets(self):

        tweet_type = "covid-19 vaccine"
        treatment = 'covid-19 vaccine'
        print("\nTreatment: covid-19 vaccine")

        # Analyzing Vaccine tweet words
        pd.set_option('display.max_columns', None)
        vaccine_df = pd.read_csv(all_tweets_filename, sep=',')
        self.vaccine_tweets_df = vaccine_df.loc[vaccine_df['treatment'] == 'covid-19 vaccine']
        self.vaccine_positive_tweets_df = self.vaccine_tweets_df.loc[self.vaccine_tweets_df['sentiment'] == 'Positive']
        self.vaccine_neutral_tweets_df = self.vaccine_tweets_df.loc[self.vaccine_tweets_df['sentiment'] == 'Neutral']
        self.vaccine_negative_tweets_df = self.vaccine_tweets_df.loc[self.vaccine_tweets_df['sentiment'] == 'Negative']
        print("Total Vaccine: ", len(self.vaccine_tweets_df['treatment']))
        print("Total Vaccine Positive: ", len(self.vaccine_positive_tweets_df))
        print("Total Vaccine Neutral: ", len(self.vaccine_neutral_tweets_df))
        print("Total Vaccine Negative: ", len(self.vaccine_negative_tweets_df))

        # Common Words for Vaccines
        new_words_in_tweets = [tweet.split() for tweet in
                               self.vaccine_tweets_df.tweet_words]  # split tweets into words
        new_words_list = list(itertools.chain(*new_words_in_tweets))  # List of all words across tweets
        new_words_list = list(filter(lambda w: w not in omit_words_list, new_words_list))  # Omit specific words
        for i in new_words_list:
            all_words_list.append(i)
        count_all_words = collections.Counter(all_words_list)  # Create counter
        split_tweets = pd.DataFrame(count_all_words.most_common(common_words_num), columns=['words', 'count'])
        print("\n\n Word Frequency for Vaccine")
        print(split_tweets.head(n=25))  # Display frequent words

        # Show tweets_words from 5 rows
        print(self.vaccine_tweets_df.head(5))

        # WordCloud for Vaccine tweets
        self.get_wordcloud(self.vaccine_tweets_df)  # Create WordCloud

        # Histogram for Vaccine tweet Sentiment
        fig, ax = plt.subplots(figsize=(8, 6))
        self.vaccine_tweets_df.polarity.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                                             ax=ax,
                                             color="purple")
        plt.title("Sentiments for 'Covid-19 Vaccine'")
        plt.show()

        # Histogram for Vaccine tweet Subjectivity
        fig, ax = plt.subplots(figsize=(8, 6))
        self.vaccine_tweets_df.subjectivity.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                                                 ax=ax,
                                                 color="purple")
        plt.title("Subjectivity for 'Covid-19 Vaccine'")
        plt.show()

        # Horizontal bar graph for Vaccine tweets
        plot_words = pd.DataFrame(count_all_words.most_common(common_words_num), columns=['words', 'count'])
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_words.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple")
        ax.set_title("Common Words in Vaccine Tweets")
        plt.show()

        # Scatter Plot sentiment for 'covid-19 vaccine'
        plot_title = "Sentiment Analysis - Covid-19 Vaccine"
        xlabel = "Polarity"
        ylabel = "Subjectivity"
        self.scatter_plot(self.vaccine_tweets_df, plot_title, xlabel, ylabel)

        # Histogram for vaccine sentiments
        plt.title('Sentiment Analysis for Covid-19 Vaccines')
        plt.xlabel('Sentiment')
        plt.ylabel('Counts')
        self.vaccine_tweets_df['sentiment'].value_counts().plot(kind='bar', color="purple")
        plt.show()

        # Analyzing and inserting for Therapeutics
        for treatment in treatments:

            tweet_type = "Therapeutics"
            new_all_words_list = []
            print("\n" + "Treatment: " + treatment)

            # Load tweets for Therapeutics
            self.therapeutics_tweets_df = pd.read_csv(all_tweets_filename, sep=',')
            self.therapeutics_tweets_df = self.therapeutics_tweets_df.loc[
                self.therapeutics_tweets_df['treatment'] == treatment]

            # Commons words for Therapeutics
            new_words_in_tweets = [tweet.split() for tweet in
                                   self.therapeutics_tweets_df.tweet_words]  # split tweets into words
            new_words_list = list(itertools.chain(*new_words_in_tweets))  # List of all words across tweets
            new_words_list = list(filter(lambda w: w not in omit_words_list, new_words_list))
            for i in new_words_list:
                all_words_list.append(i)
            count_all_words = collections.Counter(all_words_list)  # Create counter
            split_tweets = pd.DataFrame(count_all_words.most_common(common_words_num), columns=['words', 'count'])

        print("\n\n Word Frequency for Therapeutics")
        print(split_tweets.head(n=25))  # Display frequent words

        # Load tweets for therapeutics
        therapeutics_df = pd.read_csv(all_tweets_filename, sep=',')
        self.therapeutics_tweets_df = therapeutics_df.loc[therapeutics_df['treatment'] != 'covid-19 vaccine']

        self.therapeutics_positive_tweets_df = self.therapeutics_tweets_df.loc[
            self.therapeutics_tweets_df['sentiment'] == 'Positive']
        self.therapeutics_neutral_tweets_df = self.therapeutics_tweets_df.loc[
            self.therapeutics_tweets_df['sentiment'] == 'Neutral']
        self.therapeutics_negative_tweets_df = self.therapeutics_tweets_df.loc[
            self.therapeutics_tweets_df['sentiment'] == 'Negative']
        print("Total Therapeutics: ", len(self.therapeutics_tweets_df['treatment']))
        print("Total Therapeutics Positive: ", len(self.therapeutics_positive_tweets_df))
        print("Total Therapeutics Neutral: ", len(self.therapeutics_neutral_tweets_df))
        print("Total Therapeutics Negative: ", len(self.therapeutics_negative_tweets_df))

        # WordCloud for Therapeutic tweets
        self.get_wordcloud(self.therapeutics_tweets_df)  # Create WordCloud

        # Horizontal bar graph for Therapeutics
        plot_words = pd.DataFrame(count_all_words.most_common(common_words_num), columns=['words', 'count'])
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_words.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple")
        ax.set_title("Common Words in Therapeutic Tweets")
        plt.show()

        # Histogram for Therapeutic tweet sentiments
        fig, ax = plt.subplots(figsize=(8, 6))
        self.therapeutics_tweets_df.polarity.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                                                  ax=ax, color="purple")
        plt.title("Sentiments for Therapeutics")
        plt.show()

        # Histogram for Therapeutic tweet subjectivity
        fig, ax = plt.subplots(figsize=(8, 6))
        self.therapeutics_tweets_df.subjectivity.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                                                      ax=ax, color="purple")
        plt.title("Subjectivity for Therapeutics")
        plt.show()

        therapeutic_df = pd.read_csv(all_tweets_filename, sep=',')
        therapeutics_tweets_df = therapeutic_df[['treatment', 'sentiment', 'polarity', 'subjectivity', 'tweet_words']]
        all_therapeutics_tweets_df = therapeutics_tweets_df.loc[
            therapeutics_tweets_df['treatment'] != 'covid-19 vaccine']

        # Scatter Plot sentiment for Therapeutics
        # plt.figure(figsize=(8, 6))
        for i in range(1, therapeutics_df.shape[0]):
            plt.scatter(therapeutics_df["polarity"][i], therapeutics_df["subjectivity"][i], color='Purple')
        plt.title("Sentiment Analysis - Therapeutics")
        plt.xlabel("Polarity")
        plt.ylabel("Subjectivity")
        plt.show()

        # Print the percentage of negative tweets
        ntweets = self.vaccine_tweets_df[self.vaccine_tweets_df.sentiment == 'negative']
        ntweets = ntweets['tweet_words']
        print(round((ntweets.shape[0] / self.vaccine_tweets_df.shape[0]) * 100, 1))

        # Print the percentage of positive tweets
        ptweets = all_therapeutics_tweets_df[all_therapeutics_tweets_df.sentiment == 'negative']
        ptweets = ptweets['tweet_words']
        print(round((ptweets.shape[0] / all_therapeutics_tweets_df.shape[0]) * 100, 1))

        # Plotting and visualizing the counts
        plt.title('Sentiment Analysis for Covid-19 Therapeutics')
        plt.xlabel('Sentiment')
        plt.ylabel('Counts')
        all_therapeutics_tweets_df['sentiment'].value_counts().plot(kind='bar', color="purple")
        plt.show()

        # Vectorizing
        count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        vectorized_data = count_vectorizer.fit_transform(all_therapeutics_tweets_df.tweet_words)
        indexed_data = hstack((np.array(range(0, vectorized_data.shape[0]))[:, None], vectorized_data))

        # Linear SVM Classifier OneVsRestClassifier
        def sentiment2target(sentiment):
            return {
                'Negative': 'Negative',
                'Neutral': 'Neutral',
                'Positive': 'Positive'
            }[sentiment]

        # Split and train the data for SVM
        targets = all_therapeutics_tweets_df.sentiment.apply(sentiment2target)
        data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4,
                                                                              random_state=0)
        data_train_index = data_train[:, 0]
        data_train = data_train[:, 1:]
        data_test_index = data_test[:, 0]
        data_test = data_test[:, 1:]

        clf = OneVsRestClassifier(
            svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
        clf_output = clf.fit(data_train, targets_train)
        print("clf_output: ", clf_output)
        print("CLF Score: ", clf.score(data_test, targets_test))  # SVM OneVersusRest Classifier score

        # Feature Extraction
        # for twt in all_therapeutics_tweets_df.tweet_words:
        #    print("twt: ", twt)
        #    training_set = nltk.classify.apply_features(self.extract_features, twt)
        #    print(training_set[0])
        #   print(len(training_set), len(twt))

        def get_feature_vector(train_fit):
            vector = TfidfVectorizer(sublinear_tf=True)
            vector.fit(train_fit)
            return vector

        def int_to_string(sentiment):
            if sentiment == -1:
                return "Negative"
            elif sentiment == 0:
                return "Neutral"
            else:
                return "Positive"

        # Load dataset
        therapeutic_training_dataset = pd.read_csv(all_tweets_filename, sep=',')  # , encoding='latin-1'

        # Split dataset into Train, Test
        # Same tf vector will be used for Testing sentiments on unseen trending data
        tf_vector = get_feature_vector(np.array(therapeutic_training_dataset.iloc[:, 1]).ravel())
        X = tf_vector.transform(np.array(therapeutic_training_dataset.iloc[:, 1]).ravel())
        y = np.array(therapeutic_training_dataset.iloc[:, 0]).ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

        # Training Naive Bayes model
        NB_model = MultinomialNB()
        NB_model.fit(X_train, y_train)
        y_predict_nb = NB_model.predict(X_test)
        print("Naive Bayes Model Accuracy Score: ", accuracy_score(y_test, y_predict_nb))

        # Training Logistics Regression model
        LR_model = LogisticRegression(solver='lbfgs')
        LR_model.fit(X_train, y_train)
        y_predict_lr = LR_model.predict(X_test)
        print("Logistical Regression Model Accuracy: ", accuracy_score(y_test, y_predict_lr))

        # Confusion Matrix
        # Run Train Data Through Pipeline analyzer=text_process
        X_train, X_test, y_train, y_test = train_test_split(therapeutic_training_dataset['tweet_words'][:601], therapeutic_training_dataset['sentiment'][:601], test_size=0.2)
        # create pipeline
        pipeline = Pipeline([
            ('bow', CountVectorizer(strip_accents='ascii',
                                    stop_words='english',
                                    lowercase=True)),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
        ])
        # this is where we define the values for GridSearchCV to iterate over
        parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'classifier__alpha': (1e-2, 1e-3),
                      }
        # Do 10-fold cross validation for each of the 6 possible combinations of the above params
        grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
        grid.fit(X_train, y_train)
        # summarize results
        print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
        print('\n')
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        params = grid.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))

        # Accuracy, Confusion Matrix, and Classification Report
        joblib.dump(grid, "twitter_sentiment.pkl")
        # load from file and predict using the best configs found in the CV step
        model_NB = joblib.load("twitter_sentiment.pkl")
        # get predictions from best model above
        y_preds = model_NB.predict(X_test)
        print('Accuracy Score: ', accuracy_score(y_test, y_preds))
        print('\n')
        print('Confusion Matrix: \n', confusion_matrix(y_test, y_preds))
        print('\n')
        print(classification_report(y_test, y_preds))


    # def load_dataset(filename):
    #    dataset = pd.read_csv(filename, encoding='latin-1')
    #    cols = ['created_at', 'tweet_words']
    #    dataset.columns = cols
    #    return dataset

    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in w_features:
            features[f'contains({word})'] = (word in document_words)
        return features

    def scatter_plot(self, dataframe, plot_title, xlabel, ylabel):
        plt.figure(figsize=(8, 6))
        for i in range(0, dataframe.shape[0]):
            plt.scatter(dataframe["polarity"][i], dataframe["subjectivity"][i], color='Purple')
        plt.title(plot_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def get_treatment_df(self, filename):
        treatments_df = pd.read_csv(filename, sep=',')
        return treatments_df

    def get_tweet_words(self, treatment):
        treatments_df = pd.read_csv('data/TweetsBySearchTerm.csv')
        is_treatment_type = treatments_df['treatment'] == treatment
        treatments_df_type = treatments_df[is_treatment_type]  # Dataframe with tweets
        treatments_df_type.count()
        words_in_tweet = [tweet.split() for tweet in treatments_df_type.tweet_words]  # split tweets into words
        all_words = list(itertools.chain(*words_in_tweet))  # List of all words across tweets
        count_words = collections.Counter(all_words)  # Create counter
        return count_words

    def get_feature_vector(self, train_fit):
        vector = TfidfVectorizer(sublinear_tf=True)
        vector.fit(train_fit)
        return vector

    # def extract_features(word_list):
    #    return dict([(word, True) for word in word_list])

    # def create_horz_bar(self, treatment):

    # def get_common_words(self, treatment):

    def get_wordcloud(self, tweets_df):
        all_tweet_words = ' '.join([text for text in tweets_df['tweet_words']])
        from wordcloud import WordCloud
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_tweet_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # Define search variables
    treatments = ["hydroxychloroquine", "remdesivir", "regeneron"]
    csv_headers = ["created_at", "treatment", "sentiment", "polarity", "subjectivity", "text", "tweet_words"]

    omit_words_list = ['us', 'amp', 'get', 'would', 'via', 'says', 'say', 'said', 'could']

    search_filter = " -filter:retweets"
    num_tweets = 3
    common_words_num = 25
    since_date = '2020-01-01'
    all_tweets_filename = 'data/TweetsBySearchTerm.csv'
    all_words_list = []

    gt = getTweets()
    gt.process_tweets()
