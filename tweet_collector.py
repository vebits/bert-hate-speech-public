import tweepy
import csv
import json

CONSUMER_KEY = 'CONSUMER_KEY'
CONSUMER_SECRET = 'CONSUMER_SECRET'
ACCESS_TOKEN = 'ACCESS_TOKEN'
ACCESS_SECRET = 'ACCESS_SECRET'


class TweetCollector:

    def __init__(self):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


    def lookup_tweets(self, tweet_ids):
        full_tweets = []
        tweet_count = len(tweet_ids)
        try:
            for i in range((tweet_count // 100) + 1):
                end_loc = min((i + 1) * 100, tweet_count)
                full_tweets.extend(
                    self.api.statuses_lookup(id_=tweet_ids[i * 100:end_loc])
                )
            return full_tweets
        except tweepy.TweepError:
            print('Tweet, tweet, something went wrong...')
