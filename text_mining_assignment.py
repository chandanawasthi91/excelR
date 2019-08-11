import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
whey_protein=[]

'''
1) Extract reviews of any product from ecommerce website like snapdeal and amazon
2) Perform sentimental analysis
'''
#whey protein reviews
### Extracting reviews from Amazon website ################
for i in range(1,100):
  ip=[]  
  url="https://www.amazon.in/MuscleBlaze-Whey-Protein-Isolate-Chocolate/product-reviews/B071KWBBXZ/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
  whey_protein=whey_protein+ip  # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("whey_protein.txt","w") as output:
    output.write(str(whey_protein))
	
# Creating a data frame 
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

whey_protein_reviews = pd.DataFrame(columns = ["reviews"])
whey_protein_reviews["reviews"] = whey_protein

whey_protein_reviews.to_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\whey_protein_reviews.csv",encoding="utf-8",index=None,header= False)
whey_protein_rev = list(whey_protein_reviews["reviews"])


whey_protein_rev_str = ""
for i in whey_protein_rev:
    #print( i)
    whey_protein_rev_str = whey_protein_rev_str + i
    
whey_protein_rev_str = re.sub("[^A-Za-z" "]+"," ",whey_protein_rev_str).lower()
whey_protein_rev_str = re.sub("[0-9" "]+"," ",whey_protein_rev_str)

# words that contained in iphone 7 reviews
whey_protein_rev_str_words = whey_protein_rev_str.split(" ")

with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\payslip\\New folder (2)\\sw.txt","r") as sw:
    stopwords = sw.read()

#stopwords = stopwords.split("\n")
whey_protein_rev_str_words = [w for w in whey_protein_rev_str_words if not w in stopwords]


# Joinining all the reviews into single paragraph 
whey_protein_rev_str_words = " ".join(whey_protein_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(whey_protein_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\whey_protein.pdf',pad_inches=0.5,format='pdf')

wordcloud_ip_bilinear = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000
                     ).generate(whey_protein_rev_str_words)

plt.imshow(wordcloud_ip_bilinear,interpolation='bilinear')
    
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\whey_protein_bilinear.pdf',format='pdf')


#negative sentiment analyis

with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\Negative.txt","r") as neg:
    negativewords = neg.read()
negativewords = negativewords.split("\n")
whey_protein_rev_str_words = whey_protein_rev_str.split(" ")


#stopwords = stopwords.split("\n")
whey_protein_rev_str_words = [w for w in whey_protein_rev_str_words if not w in stopwords]
whey_protein_rev_str_words = [w for w in whey_protein_rev_str_words if w in negativewords]


# Joinining all the reviews into single paragraph 
whey_protein_rev_str_words = " ".join(whey_protein_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(whey_protein_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\whey_protein_negative.pdf',pad_inches=0.5,format='pdf')

#positive sentiment analyis

with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\Positive.txt","r") as neg:
    positiveewords = neg.read()
positiveewords = positiveewords.split("\n")
whey_protein_rev_str_words = whey_protein_rev_str.split(" ")

#stopwords = stopwords.split("\n")
whey_protein_rev_str_words = [w for w in whey_protein_rev_str_words if not w in stopwords]
whey_protein_rev_str_words = [w for w in whey_protein_rev_str_words if w in positiveewords]


# Joinining all the reviews into single paragraph 
whey_protein_rev_str_words = " ".join(whey_protein_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(whey_protein_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\whey_protein_positive.pdf',pad_inches=0.5,format='pdf')

'''
1) Extract movie reviews for any movie from IMDB and perform sentimental analysis
2) Extract anything you choose from the internet and do some research on how we extract using R/Python
   Programming and perform sentimental analysis.
'''


from selenium import webdriver
import os
chromedriver = "C:/Users/cawasthi/Downloads/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chromedriver
browser = webdriver.Chrome(chromedriver) # opens the chrome browser
from bs4 import BeautifulSoup as bs

## Moana Movie #####
page= "https://www.imdb.com/title/tt7485048/reviews?ref_=tt_ql_3"
# Importing few exceptions to surpass the error messages while extracting reviews 
from selenium.common.exceptions import NoSuchElementException 
from selenium.common.exceptions import ElementNotVisibleException

browser.get(page)
import time
reviews = []
i=1
# Below while loop is to load all the reviews into the browser till load more button dissapears
while (i>26):
	i=i+1
	try:
		# Storing the load more button page xpath which we will be using it for click it through selenium 
		# for loading few more reviews
		button = browser.find_element_by_xpath('//*[@id="text show-more__control"]') # //*[@id="load-more-trigger"]
		button.click()
		time.sleep(5)
	except NoSuchElementException:
		break
	except ElementNotVisibleException:
		break

# Getting the page source for the entire imdb after loading all the reviews
ps = browser.page_source 
#Converting page source into Beautiful soup object
soup=bs(ps,"html.parser")

#Extracting the reviews present in div html_tag having class containing "text" in its value
reviews = soup.findAll("div",attrs={"class","text"})
for i in range(len(reviews)):
	reviews[i] = reviews[i].text
    
# Creating a data frame 
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

super_30_reviews = pd.DataFrame(columns = ["reviews"])
super_30_reviews["reviews"] = reviews

super_30_reviews.to_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\super_30_reviews.csv",encoding="utf-8",index=None,header= False)
super_30_rev = list(super_30_reviews["reviews"])


super_30_rev_str = ""
for i in super_30_rev:
    #print( i)
    super_30_rev_str = super_30_rev_str + i
    
super_30_rev_str = re.sub("[^A-Za-z" "]+"," ",super_30_rev_str).lower()
super_30_rev_str = re.sub("[0-9" "]+"," ",super_30_rev_str)

# words that contained in iphone 7 reviews
super_30_rev_str_words = super_30_rev_str.split(" ")


with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\payslip\\New folder (2)\\sw.txt","r") as sw:
    stopwords = sw.read()

#stopwords = stopwords.split("\n")
super_30_rev_str_words = [w for w in super_30_rev_str_words if not w in stopwords]


# Joinining all the reviews into single paragraph 
super_30_rev_str_words = " ".join(super_30_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(super_30_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\super_30.pdf',pad_inches=0.5,format='pdf')

wordcloud_ip_bilinear = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000
                     ).generate(super_30_rev_str_words)

plt.imshow(wordcloud_ip_bilinear,interpolation='bilinear')
    
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\super_30_bilinear.pdf',format='pdf')


#negative sentiment analyis

with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\Negative.txt","r") as neg:
    negativewords = neg.read()
negativewords = negativewords.split("\n")
super_30_rev_str_words = super_30_rev_str.split(" ")

#stopwords = stopwords.split("\n")
super_30_rev_str_words = [w for w in super_30_rev_str_words if not w in stopwords]
super_30_rev_str_words = [w for w in super_30_rev_str_words if w in negativewords]


# Joinining all the reviews into single paragraph 
super_30_rev_str_words = " ".join(super_30_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(super_30_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\super_30_negative.pdf',pad_inches=0.5,format='pdf')

#positive sentiment analyis

with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\Positive.txt","r") as neg:
    positiveewords = neg.read()
positiveewords = positiveewords.split("\n")
super_30_rev_str_words = super_30_rev_str.split(" ")

#stopwords = stopwords.split("\n")
super_30_rev_str_words = [w for w in super_30_rev_str_words if not w in stopwords]
super_30_rev_str_words = [w for w in super_30_rev_str_words if w in positiveewords]


# Joinining all the reviews into single paragraph 
super_30_rev_str_words = " ".join(super_30_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(super_30_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\super_30_positive.pdf',pad_inches=0.5,format='pdf')

'''
 ONE:
1) Extract tweets for any user (try choosing a user who has more tweets)
2) Perform sentimental analysis on the tweets extracted from the above
'''

import re
import os
print(os.getcwd())
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

ACCESS_TOKEN = "1157869137638510593-c5WqkUMYaMJZtxGV5Ww9bfLWDdcaPd"
ACCESS_SECRET_TOKEN = "f5pvx2partvnzD7eet7YgRLjSVSTOJKEh13mvUYHMN8AD"
CONSUMER_KEY = "PL3li1NjCupgmhMZIA06N2ccO"
CONSUMER_SECRET = "DcnKFVjz81jWsA5vhM0X938EviwTMzamjE9MG0RSpD3VBSTVYY"

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
    tweets = api.user_timeline(screen_name='narendramodi', count = 500)

    list_of_tweet_texts = [tweet.text for tweet in tweets ]
    #print(list_of_tweet_texts)
	
tweet_df = pd.DataFrame(columns=["reviews"])
tweet_df["reviews"] = list_of_tweet_texts


tweet_df_rev_str = ""
for i in list_of_tweet_texts:
    #print( i)
    tweet_df_rev_str = tweet_df_rev_str + i

#tweet_df.to_csv("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\tweets.csv",encoding="utf-8")

tweet_df_rev_str = re.sub("[^A-Za-z" "]+"," ",tweet_df_rev_str).lower()
tweet_df_rev_str = re.sub("[0-9" "]+"," ",tweet_df_rev_str)
#print(tweet_df_rev_str)


# words that contained in iphone 7 reviews
tweet_df_rev_str_words = tweet_df_rev_str.split(" ")


with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\payslip\\New folder (2)\\sw.txt","r") as sw:
    stopwords = sw.read()

#stopwords = stopwords.split("\n")
tweet_df_rev_str_words = [w for w in tweet_df_rev_str_words if not w in stopwords]


# Joinining all the reviews into single paragraph 
tweet_df_rev_str_words = " ".join(tweet_df_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=50,margin=10,random_state=1
                     ).generate(tweet_df_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\tweet.pdf',pad_inches=0.5,format='pdf')


#negative sentiment analyis


with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\Negative.txt","r") as neg:
    negativewords = neg.read()
negativewords = negativewords.split("\n")
#print(negativewords)
tweet_df_rev_str_words = tweet_df_rev_str.split(" ")

#stopwords = stopwords.split("\n")

tweet_df_rev_str_words = [w for w in tweet_df_rev_str_words if w in negativewords]
print(tweet_df_rev_str_words)
# Joinining all the reviews into single paragraph 
tweet_df_rev_str_words = " ".join(tweet_df_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=25,margin=10,random_state=1
                     ).generate(tweet_df_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\negative.pdf',pad_inches=0.5,format='pdf')



#positive sentiment analyis

with open("C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\Positive.txt","r") as neg:
    positiveewords = neg.read()
positiveewords = positiveewords.split("\n")
tweet_df_rev_str_words = tweet_df_rev_str.split(" ")

#stopwords = stopwords.split("\n")
tweet_df_rev_str_words = [w for w in tweet_df_rev_str_words if w in positiveewords]


# Joinining all the reviews into single paragraph 
tweet_df_rev_str_words = " ".join(tweet_df_rev_str_words)

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=2500,
                      height=2000,max_words=25,margin=10,random_state=1
                     ).generate(tweet_df_rev_str_words)

plt.imshow(wordcloud_ip)
plt.axis("off")
plt.savefig('C:\\Users\\cawasthi\\Desktop\\Data Science\\R ML Code\\text mining\\positive.pdf',pad_inches=0.5,format='pdf')

