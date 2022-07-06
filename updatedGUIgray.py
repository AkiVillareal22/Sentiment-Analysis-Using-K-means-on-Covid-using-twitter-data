import tkinter
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from tkinter.tix import *
from pickle import TRUE

import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import random
import string
import time
import fileinput
import math
import io

import pandas as pd
import tweepy
from tweepy import OAuthHandler
from nltk.corpus.reader import wordlist
import numpy as np
from numpy import savetxt
import re
import nltk
from nltk.tokenize import RegexpTokenizer, regexp
import csv
from collections import Counter
from csv import reader
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from numpy import random
from pylab import plot, show
import seaborn as sns

#########################################################################################


num_rows_fileData = 10  # Initialization of variables
num_col_fileData = 2
keyword = ''
lemmatizer = WordNetLemmatizer()
folder_or_file = ''
v = ''

################################################################################################################
consumer_key = "V2gVQljCb6XWQBw1qbkD3vL46"
consumer_secret = "ptcI50s0BRSQYNaOriPkBDTmVPq7Sd9yX6uGsP9GF9kVv7bgZv"  # TWITTER API
access_token = "1456202874778619911-LikiUZSzvxOcsUk6pcDQRUNpcKbr2l"
access_token_secret = "LdwpGiSq5Iid6ITDMO5cZNsi84sH7YoDzdnqeaPgOPK1a"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

######################################################


def cleanTxt(text):
    #removing @mentions
    # r tells python the the re is raw string
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # removing hashtags
    text = re.sub(r'#', '', text)
    # removing RT
    text = re.sub(r'RT[\s]+', '', text)
    # removing urls
    text = re.sub(r'http\S+', '', text)
    # removing colon
    text = re.sub(r':', '', text)
    # removing â€™
    text = re.sub(r'_', '', text)
    # removing symbols â€™
    text = re.sub(
        "[^-_a-zA-Z0-9!@#%&=,/'\";:~`\$\^\*\(\)\+\[\]\.\{\}\|\?\<\>\\]+|[^\s]+", '', text)
    # removing $
    text = re.sub(r'[$]', '', text)
    # removing dot
    text = re.sub(r'[.]', '', text)
    # removing comma
    text = re.sub(r'[,;]', '', text)
    # removing parenthesis
    text = re.sub(r'[()]', '', text)
    # removing dash
    text = re.sub(r'[-]', '', text)
    #removing &amp
    text = re.sub(r'[&]', '', text)
    # removing 0-9
    text = re.sub(r'[0-9]', '', text)
    # removing ""
    text = re.sub(r'["]', '', text)
    # removing /\r?\n|\r/
    text = re.sub(r'[/\r?\n|\r/]', '', text)

    return text


def Convert(string):
    #will receive a string converting it to a list of string
    li = list(string.split(" "))
    return li

def remove_values_from_list(the_list, val):
    #will remove other values from the list other than the passed value
    return [value for value in the_list if value != val]


def extractTweets():
    with open("twitterData2.csv", "w", newline="", encoding="utf-8") as f:
        #will get a keyword from the user then interact with tweeter api to get tweets
        keyword = enterKeyWordField.get()
        if keyword == '':
            #will make sure the keyword has a value
            messagebox.showerror(
                "Error", "Please Enter Keyword on the Text Field")

        if keyword != '':
            writer = csv.writer(f)
            queryWord = keyword
            print("Your query word is: ", queryWord)
            messagebox.showinfo(
                "Message", "Obtaining Tweets with the Selected Keyword")
            cursor = tweepy.Cursor(api.search_tweets, q=queryWord,
                                   geocode="14.58012,121.02225,50km",
                                   tweet_mode="extended").items(200)
            for index, tweet in enumerate(cursor):
                data = (index, tweet.full_text)
                writer.writerow(data)
        f.close()
        enterKeyWordField.delete(0, 'end')


# openfile function
def Table_Window():
    #   win2 = Toplevel()
    #   message = "Table"
    #   Label(win2, text=message).pack()
    #   # Adjusting size
    #   win2.geometry("600x500")
    #   # Minimum window size
    #   win2.minsize(400, 300)
    #   # Maximum window size
    #   win2.maxsize(600, 500)

    #   my_tree2 = ttk.Treeview(win2)

    # ============================================================
    fileName = filedialog.askopenfilename(initialdir="E:\Study Files\Python Study'\Thesis Study",
                                          title="Select A File",
                                          filetypes=(("csv files", "*.csv"), ("all files", "*.*")))

    v.set(fileName)
    folder_or_file = str(v.get())
    print(folder_or_file)
    os.path.split(fileName)
    fileText.set(os.path.split(fileName)[1])

    try:
        fileName = r"{}".format(fileName)
        df = pd.read_csv(fileName, names=["Index","Tweets"] )

      #  win2 = Toplevel()
      #  message = "Table"
      #  Label(win2, text=message).pack()
        # Adjusting size
      #  win2.geometry("900x650")
        # Minimum window size
      #  win2.minsize(600, 600)
        # Maximum window size
      #  win2.maxsize(900, 650)
      #  style = ttk.Style(win2)
      #  style.configure('Treeview', rowheight=40)

       #####edit

    except ValueError:
        my_label.config(text="File Couldn't be Opened.")
    except FileNotFoundError:
        my_label.config(text="File Couldn't be Opened.")

    clear_tree()

    my_tree2["column"] = list(df.columns)
    my_tree2["show"] = "headings"
    # looping through column list
    for column in my_tree2["column"]:
        my_tree2.heading(column, text=column)

    # filling treeview with data
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        my_tree2.insert("", "end", values=row)

    my_tree2.pack(expand=True, fill='both')


def clear_tree():
    my_tree2.delete(*my_tree2.get_children())



################################################################################
def open_window():
    ################### EMOTICON #########################
    def emoticon():
        #will check the emotion present on the tweets 
        emotion_list = []
        word_list = []
        with open('processedTweets.csv', 'r') as clean_tweets:
            read_tweets = reader(clean_tweets)
            lister = list(read_tweets)
            
            for row in lister:
                regexp_tokenizer = RegexpTokenizer("[\w']+")
                new_word_token = regexp_tokenizer.tokenize(row[1])
                word_list = new_word_token
                # will compare the words to the bagofwords of the emotion text
                with open('emotion.txt', 'r') as emote_file:
                    for line in emote_file:
                        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                        word, emotion = clear_line.split(':')
                       
                        if word in word_list:
                            emotion_list.append(emotion)

        print(lister, emotion_list)
        w = Counter(emotion_list)
        print(w)
        fig, axl = plt.subplots()
        axl.bar(w.keys(), w.values())
        fig.autofmt_xdate()
        plt.show()

    ################### EMOTICON #########################

    ###################### SENTIMENT ANALYSIS #####################
    def sentimentAnalysis():
        # will check the sentiment of the processed tweets with the help of the TFIDF function to filter the tweets 
        df = pd.read_csv('processedTweets.csv', header=None,encoding="unicode_escape")
        df.drop_duplicates((1), inplace=True)
        df.to_csv('processedTweets.csv', index=False)
        clear_tree()
        classification = []
        # will compare the words of each tweets to the bag of words
        f = open("sentiment.csv", 'w', newline='')

        with open('negativeBagOfWords.csv', newline='') as negBag:
            reader = csv.reader(negBag)
            negBagData = list(reader)

        with open('positveBagOfWords.csv', newline='') as posBag:
            reader = csv.reader(posBag)
            posBagData = list(reader)

        with open('inverseWordDatabag.csv', newline='') as inverseBag:
            reader = csv.reader(inverseBag)
            inverseBagData = list(reader)

        with open("processedTweets.csv", newline="") as trimSen:
            reader2 = csv.reader(trimSen)
            trimSenData = list(reader2)

            
        #will tally the score of each tweets for its polarity
            for data in trimSenData:
                stringListSentence = Convert(data[1])
                
                sentenceScore = 0
                for negData in negBagData:
                    if negData[1] not in stringListSentence:
                        sentenceScore += 0
                    elif negData[1] in stringListSentence:
                        index = stringListSentence.index(negData[1])
                        prevWordList = [stringListSentence[index - 1]]
                        if prevWordList in inverseBagData:
                            sentenceScore += 1
                        else:
                            sentenceScore -= 1
                for posData in posBagData:
                    if posData[1] not in stringListSentence:
                        sentenceScore += 0
                    elif posData[1] in stringListSentence:
                        index = stringListSentence.index(posData[1])
                        prevWordList = [stringListSentence[index - 1]]
                        if prevWordList in inverseBagData:
                            sentenceScore -= 1
                        else:
                            sentenceScore += 1
                            

                if sentenceScore < 0:
                    classification = "Negative"
                elif sentenceScore == 0:
                    classification = "Neutral"
                elif sentenceScore > 0:
                    classification = "Positive"
                  
                

                sentiment = (data[1], classification)
                print(sentiment)
                f.write(str(sentiment) + "\n")

                
                sentiment = r"{}".format(sentiment)
                data = io.StringIO(sentiment)
                df1 = pd.read_csv(data, sep=",", names=["Tweet", "Classification"])



                my_tree3["column"] = list(df1.columns)
                my_tree3["show"] = "headings"
                # looping through column list
                for column in my_tree3["column"]:
                    my_tree3.heading(column, text=column)

                df_rows = df1.to_numpy().tolist()
                for row in df_rows:
                    my_tree3.insert("", "end", values=row)

                my_tree3.pack(expand=True, fill='both')


            ###################### SENTIMENT ANALYSIS #####################

    ######################### PROCESSING OF TWEETS #######################
    def processTweets():
        # Pre processing the tweets removing any noise words and stop words
        with open('twitterData2.csv', newline='', encoding="utf-8") as tweetsData:
            reader = csv.reader(tweetsData)
            tweets = list(reader)

        with open('eng_stop_words.csv', newline='', encoding="utf-8") as engStop:
            reader2 = csv.reader(engStop)
            engStopData = list(reader2)

        with open('fil_stop_words.csv', newline='', encoding="utf-8") as filStop:
            reader3 = csv.reader(filStop)
            filStopData = list(reader3)

        with open("processedTweets.csv", 'w', newline="") as process:

            count = 0
            for tweet in tweets:
                count += 1
                cleanTweet = cleanTxt(tweet[1].lower())
                cleanTweettoList = Convert(cleanTweet.lower())
                word_list = []
                for word in cleanTweettoList:
                    for dataEng in engStopData:
                        if dataEng[1] == word:
                            cleanTweettoList = remove_values_from_list(
                                cleanTweettoList, word)
                    for dataFil in filStopData:
                        if dataFil[1] == word:
                            cleanTweettoList = remove_values_from_list(
                                cleanTweettoList, word)
                for word in cleanTweettoList:
                    word_list.append(lemmatizer.lemmatize(word))
                    sentence = " ".join(word_list)
                processedTweet = " ".join(cleanTweettoList)

                data_tuple = (count, processedTweet)
                writer = csv.writer(process)
                writer.writerow(data_tuple)

        messagebox.showinfo("Message", "Processed tweets -- Removed Stop Words -- Created new csv file")
        process.close()
        engStop.close()
        filStop.close()
        tweetsData.close()

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def remove_space():
        df = pd.read_csv('tfidfClean.csv')
        df.to_csv('tfidfClean_NoSpace.csv', index=False)

    def processKmeans():
        K = 3  

        df = pd.read_csv('tfidfClean.csv')

        df.head()

        plt.scatter(df['frequency'], df['TFIDF'])
        plt.xlabel('Frequency')
        plt.ylabel('TF - IDF')

        km = KMeans(n_clusters=3)
        y_predicted = km.fit_predict(df[['frequency', 'TFIDF']])
        df['cluster'] = y_predicted

        df1 = df[df.cluster == 0]
        df2 = df[df.cluster == 1]
        df3 = df[df.cluster == 2]

        plt.scatter(df1['frequency'], df1['TFIDF'], color='green')
        plt.scatter(df2['frequency'], df2['TFIDF'], color='red')
        plt.scatter(df3['frequency'], df3['TFIDF'], color='black')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
                                               :, 1], color='purple', marker='*', label='centroid')
        plt.legend()
        plt.show()

        df.to_csv('TFIDF_Cluster.csv')

    def TFIDF_Cluster():
        #TF-IDF clustering for topic labeling
        clustertemp = 0
        sentenceOccuredHigh1 = 0
        sentenceOccuredHigh2 = 0
        sentenceOccuredHigh3 = 0

        topic_list = []
        topicNum_Occurence = []
        with open('TFIDF_Cluster.csv', newline='') as tfidfCL:
            reader = csv.DictReader(tfidfCL)
            cleanTFIDF = list(reader)
            #will open the TFIDF_Cluster csv to get the topic of the over all document(200 tweets)
            for line in cleanTFIDF:
                # clustering the topic label based on the sentence occurence of that topic
                if int(line['cluster']) > clustertemp:
                    clustertemp = int(line['cluster'])
            for line in cleanTFIDF:
                if int(line['cluster']) == clustertemp:
                    if int(float(line['SentenceOccured'])) > sentenceOccuredHigh1:
                        sentenceOccuredHigh1 = int(float(line['SentenceOccured']))
            for line in cleanTFIDF:
                if int(line['cluster']) == clustertemp:
                    if int(float(line['SentenceOccured'])) > sentenceOccuredHigh2 and not int(
                            float(line['SentenceOccured'])) >= sentenceOccuredHigh1:
                        sentenceOccuredHigh2 = int(float(line['SentenceOccured']))
            for line in cleanTFIDF:
                if int(line['cluster']) == clustertemp:
                    if int(float(line['SentenceOccured'])) > sentenceOccuredHigh3 and not int(
                            float(line['SentenceOccured'])) >= sentenceOccuredHigh1 and not int(
                        float(line['SentenceOccured'])) >= sentenceOccuredHigh2:
                        sentenceOccuredHigh3 = int(float(line['SentenceOccured']))
            for line in cleanTFIDF:
                if int(float(line['SentenceOccured'])) == sentenceOccuredHigh1 and len(topic_list) < 5:
                    topic_list.append(line['word'])
                    topicNum_Occurence.append(int(float(line['SentenceOccured'])))
            for line in cleanTFIDF:
                if int(float(line['SentenceOccured'])) == sentenceOccuredHigh2 and len(topic_list) < 5:
                    topic_list.append(line['word'])
                    topicNum_Occurence.append((int(float(line['SentenceOccured']))))
            for line in cleanTFIDF:
                if int(float(line['SentenceOccured'])) == sentenceOccuredHigh3 and len(topic_list) < 5:
                    topic_list.append(line['word'])
                    topicNum_Occurence.append((int(float(line['SentenceOccured']))))

            

            topic_list = r"{}".format(topic_list)
            data = io.StringIO(topic_list)
            df1 = pd.read_csv(data, sep=",", names=["1st", "2nd", "3rd", "4th"])

            clear_tree()
        
          

            my_tree3["column"] = list(df1.columns)
            my_tree3["show"] = "headings"
            # looping through column list
            for column in my_tree3["column"]:
                my_tree3.heading(column, text=column)

            df_rows = df1.to_numpy().tolist()
            for row in df_rows:
                my_tree3.insert("", "end", values=row)

            my_tree3.column("# 1", anchor=CENTER, stretch=NO, width=100)
            my_tree3.column("# 2", anchor=CENTER, stretch=NO, width=100)
            my_tree3.column("# 3", anchor=CENTER, stretch=NO, width=100)
            my_tree3.column("# 4", anchor=CENTER, stretch=NO, width=100)
     

        tfidfCL.close()

    def clear_tree():
        my_tree3.delete(*my_tree3.get_children())

    def frequency():
        #this function will compute for the TFIDF scores that will be used on the sentiment analysis and topic labeling
        with open("processedTweets.csv", newline='') as processedTweets:
            readerTweets = csv.reader(processedTweets)
            processData = list(readerTweets)

        bagOfWords = []
        for sentence in processData:
            wordsList = Convert(sentence[1])
            for word in wordsList:
                if word.strip() == "":
                    continue
                else:
                    bagOfWords.append(word.lower())

        fieldNames = ['word', 'frequency', 'SentenceOccured', 'TFIDF']
        with open("tfidf.csv", 'w', newline='') as tfidf:
            writer = csv.DictWriter(tfidf, fieldnames=fieldNames)
            writer.writeheader()
            #Nested loop that goes in and out the documents(tweets) 
            # to compute for the TF(first) then the (IDF) to get the TF-IDF scores
            for sentence in processData:
                wordsList = Convert(sentence[1])
                for word in wordsList:
                    word_frequency = 0
                    tf = 0
                    sentenceContaining_word = 0
                    idf = 0
                    tf_idf = 0
                    for x in bagOfWords:
                        if x == word:
                            word_frequency += 1

                    for sentence in processData:
                        wordsList = Convert(sentence[1])
                        if word in wordsList:
                            sentenceContaining_word += 1

                    tf = word_frequency / len(bagOfWords)
                    temp = len(processData) / sentenceContaining_word
                    idf = math.log(temp)
                    tf_idf = tf * idf
                    result = (word, word_frequency, sentenceContaining_word, tf_idf)
                    writer = csv.DictWriter(tfidf, fieldnames=fieldNames)
                    writer.writerow({'word': word, 'frequency': word_frequency,
                                     'SentenceOccured': sentenceContaining_word, 'TFIDF': tf_idf})

        seen = set()  # set for fast O(1) amortized lookup
        for line in fileinput.FileInput('tfidf.csv', inplace=1):
            if line in seen:
                continue  # skip duplicate

            seen.add(line)
            print(line)  # standard output is now redirected to the file

        with open('tfidf.csv') as in_file:
            csv_reader = csv.reader(in_file)
            with open('tfidfClean.csv', 'w') as out_file:
                writer = csv.writer(out_file)
                for line in csv_reader:
                    if line:
                        writer.writerow(line)
                    elif any(line):
                        writer.writerow(line)
                    elif any(field.strip() for field in line):
                        writer.writerow(line)

        df = pd.read_csv('tfidf.csv', encoding='unicode_escape')
        df = df[df['frequency'].notna()]
        df.to_csv('tfidfClean.csv', index=False)

        messagebox.showinfo("Message", "Successfully scored the words in the tweets")

        in_file.close()
        out_file.close()
        processedTweets.close()
        tfidf.close()

    def emoticon_and_kmeans():
        emoticon()
        processKmeans()

    def sequence(*functions):
        def func(*args, **kwargs):
            return_value = None
            for function in functions:
                return_value = function(*args, **kwargs)
            return return_value

        return func

#GUI for 2nd Window
    top = Toplevel()
    top.title('Sentiment Analysis on Social Media')
    top.configure(bg = "#1f2839")
    top.geometry("800x500")
    canvas = Canvas(
        top,
        bg = "#1f2839",
        height = 500,
        width = 800,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge")
    canvas.place(x = 0, y = 0)
    frame2 = tk.LabelFrame(top, text="Results", bg="#f5f5ef")
    frame2.place(
        x=298, y=92,
        width=453,
        height=321)
    my_tree3 = ttk.Treeview(frame2)
    my_tree3.place(relheight=1, relwidth=1)
    style = ttk.Style(top)
    style.configure('Treeview',
                    background="#f5f5ef",
                    fg="#765D69",
                    rowheight=25,
                    fieldbackground="#FEFAD4")
    style.map('Treeview',
              background=[('selected', '#615629')])
    treescrolly = Scrollbar(frame2, orient="vertical", command=my_tree3.yview)
    treescrollx = Scrollbar(frame2, orient="horizontal", command=my_tree3.xview)
    my_tree3.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y")

    # Processing of Tweets
    # tooltip for processTweetsButton
    processTip = Balloon(top)
    processButton = PhotoImage(file='buttons/preprocess.png')
    canvas.image= processButton
    processTweetsButton = Button(top, image=canvas.image, borderwidth=0, command=processTweets)
    processTweetsButton.config(bg="#1f2839", activebackground="#1f2839")
    processTweetsButton.place(
        x = 51, y = 43)
    processTip.bind_widget(processTweetsButton, balloonmsg="Preprocess the gathered data")

    # Frequency
    # tooltip for freqButton
    freqTip = Balloon(top)
    freqButton1 = PhotoImage(file='buttons/freq.png')
    canvas.image1 = freqButton1
    freqButton = Button(top, image=canvas.image1, borderwidth=0, command=frequency)
    freqButton.config(bg="#1f2839", activebackground="#1f2839")
    freqButton.place(
        x=51, y=115)
    freqTip.bind_widget(freqButton, balloonmsg="Scoring of data")


    # Process K Means
    # tooltip for K-means
    kmeansTip = Balloon(top)
    kmeansButton = PhotoImage(file='buttons/kmeans.png')
    canvas.image2 = kmeansButton
    processmeansButton = Button(top, image=canvas.image2, borderwidth=0,command=processKmeans)
    processmeansButton.config(bg="#1f2839", activebackground="#1f2839")
    processmeansButton.place(
        x=51, y=187
        )
    kmeansTip.bind_widget(processmeansButton, balloonmsg="Processing of K-means by graph")

    # Analysis
    # tooltip for Sentiment Analysis
    sentiTip = Balloon(top)
    sentiButton = PhotoImage(file='buttons/sentiment.png')
    canvas.image3 = sentiButton
    sentimentAnalysisButton = Button(top, image=canvas.image3, borderwidth=0,command=sentimentAnalysis)
    sentimentAnalysisButton.config(bg="#1f2839", activebackground="#1f2839")
    sentimentAnalysisButton.place(
        x = 51, y = 259)
    sentiTip.bind_widget(sentimentAnalysisButton, balloonmsg="Sentiment Analysis of streamed data")


    # Emotion
    # tooltip for Emotion
    emoTip = Balloon(top)
    emoButton = PhotoImage(file='buttons/emotion.png')
    canvas.image4 = emoButton
    emoticonButton = Button(top, image=canvas.image4, borderwidth=0, command=emoticon)
    emoticonButton.config(bg="#1f2839", activebackground="#1f2839")
    emoticonButton.place(
        x = 51, y = 331)
    emoTip.bind_widget(emoticonButton, balloonmsg="Emotion Detection")


    # TopicLabel || TFIDF_CLuster
    # tooltip for Sentiment Analysis
    topicTip = Balloon(top)
    labelButton = PhotoImage(file='buttons/topic.png')
    canvas.image5 = labelButton
    topicButton = Button(top, image=canvas.image5, borderwidth=0, command=TFIDF_Cluster)
    topicButton.config(bg="#1f2839", activebackground="#1f2839")
    topicButton.place(
        x = 51, y = 403)
    topicTip.bind_widget(topicButton, balloonmsg="Overall Topic Detection")



#GUI for First Window
root = Tk()
v = tk.StringVar()

#specifications for root window
#and Canvas
root.geometry("800x600")
root.title('Sentiment Analysis on Social Media')
root.configure(bg = "#1f2839")
canvas = Canvas(
    root,
    bg = "#1f2839",
    height = 600,
    width = 800,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

# Creating a tuple containing
# the specifications of the font.
Font_tuple = ("Poppins", 20)

#Keyword Field for Searching keywords
enterKeyWordField = Entry(root, font = ("Inter",12), width=30, bg="#f5f5ef", fg="#030e12")
enterKeyWordField.insert(END, 'Enter Keyword')
enterKeyWordField.bind("<FocusIn>", lambda args: enterKeyWordField.delete('0', 'end'))
enterKeyWordField.place(
    x = 120, y = 60,
    width = 330,
    height = 35)

#Stream Tweets Button and
#tooltip for streamTweetsButton
streamTip = Balloon(root)
streamButton = PhotoImage(file='buttons/Stream.png')
streamTweetsButton = Button(root, image=streamButton, borderwidth=0, command=extractTweets)
streamTweetsButton.config(bg="#1f2839",activebackground="#1f2839")
streamTweetsButton.place(
    x= 500, y=45
)
streamTip.bind_widget(streamTweetsButton, balloonmsg="Real-time streaming of data using the Keyword")


#Open File Button and
#tooltip for openButton
openTip = Balloon(root)
openButton = PhotoImage(file='buttons/Browse.png')
openFileButton = Button(root, image=openButton, borderwidth=0, command=Table_Window)
openFileButton.place(
    x= 500, y=125)
openFileButton.config(bg="#1f2839", activebackground="#1f2839")
openTip.bind_widget(openFileButton, balloonmsg="Opening of generated file after streaming ")

fileText= StringVar()
fileNameLabel = Label(root, text='', font =('Open Sans', 12), width=30, textvariable = fileText, bg="#f5f5ef", fg="#030e12")
fileNameLabel.place(
    x = 120, y = 140,
    width = 330,
    height = 35)

#Next Button and
#tooltip for nextButton
nextTip = Balloon(root)
nextButton1 = PhotoImage(file='buttons/next.png')
nextButton = Button(root, image=nextButton1, borderwidth=0, command=open_window)
nextButton.place(
    x = 500, y = 520)
nextButton.config(bg="#1f2839", activebackground="#1f2839")
nextTip.bind_widget(nextButton, balloonmsg="Proceeds to the next Window")

#BrowsFiles TreeView
frame1= tk.LabelFrame(root,text="Tweets",bg="#f5f5ef")
frame1.place(
    x = 120, y = 215,
    width= 570,
    height= 300
)

my_tree2 = ttk.Treeview(frame1)
my_tree2.place(relheight=1, relwidth=1)

style= ttk.Style()
style.configure('Treeview',
                background="#f5f5ef",
                fg="#eeded7",
                rowheight=25,
                fieldbackground="#eeded7")
style.map('Treeview',
          background=[('selected', '#615629')])
treescrolly = Scrollbar(frame1, orient="vertical", command=my_tree2.yview)
treescrollx = Scrollbar(frame1, orient="horizontal", command=my_tree2.xview)
my_tree2.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
treescrollx.pack(side="bottom", fill="x")
treescrolly.pack(side="right", fill="y")

my_label = Label(root, bg = '#8FB9A8')
my_label.grid()

root.resizable(False, False)
root.mainloop()
