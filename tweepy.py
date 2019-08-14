
import os
print(os.getcwd())

ACCESS_TOKEN = ""
ACCESS_SECRET_TOKEN = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""

#import twitter_credentials as tc
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import Cursor
from tweepy import API
import time
import pandas as pd
import numpy as np

class TwitterClient():
	def __init__(self,twitter_user=None):
		auth = TwitterAuthorization().authorizeTwitter()
		self.twitterClient = API(auth)
		self.twitter_user = twitter_user
	
	def get_twitter_client(self):
		return self.twitterClient
	
	def read_timeline_tweets(self,num_of_tweets):
		tweets = []
		for tweet in Cursor(self.twitterClient.user_timeline,id=self.twitter_user).items(num_of_tweets):
			tweets.append(tweet)
		return tweets

	def get_friends(self,num_of_friends):
		friend_list = []
		for friend in Cursor(self.twitterClient.friends).items(num_of_friends):
			friend_list.append(friend)
		return friend_list
    
class TwitterAuthorization:
	def authorizeTwitter(self):
		auth = OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
		auth.set_access_token(ACCESS_TOKEN,ACCESS_SECRET_TOKEN)
		return auth
    
class TwitterStreamer:
	def twitter_stream_tweets(self,hash_tags,file_name):
		listener = StdOutListener(file_name)
		TwitterAuthorizationobj = TwitterAuthorization()
		auth = TwitterAuthorizationobj.authorizeTwitter()
		stream = Stream(auth,listener)
		stream.filter(track=hash_tags)

class StdOutListener(StreamListener):
	def __init__(self,tweet_file_name):
		self.tweet_file_name=tweet_file_name
	
	def on_data(self,data):
		try:
			with open(self.tweet_file_name,'a') as tf:
				tf.write(data)
			return True
		except BaseException as e:
			print("Error in data %s" % str(e))
		return True
	
	def on_error(self,status):
		#Return false and terminate connection in rate limit occurs
		print(status)
		if status == 420:
			return False
		pass

if __name__ == "__main__":
    twitterclient = TwitterClient()
    api = twitterclient.get_twitter_client()
    tweets = api.user_timeline(screen_name='ChandanAwasth91', count = 5)

    list_of_tweet_texts = [tweet.text for tweet in tweets ]
    print(list_of_tweet_texts)
