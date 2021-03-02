import json
import csv
import glob
import pandas as pd
import re
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from collections import OrderedDict
from tweet_collector import TweetCollector

nltk.download('stopwords')

founta_label_folder_mapping = {
    'abusive': 'datasets/founta/offensive',
    'hateful': 'datasets/founta/hateful',
    'normal': 'datasets/founta/normal'
}

davidson_label_folder_mapping = {
    '0': 'datasets/davidson/hateful',
    '1': 'datasets/davidson/offensive',
    '2': 'datasets/davidson/normal'
}


def structure_founta(tweet_ids):
    with open('data.json', 'r') as f:
        data = json.load(f)
        no_of_spam = 0
        for i, tweet_obj in enumerate(data):
            try:
                label = tweet_ids[tweet_obj['id']]
                folder = founta_label_folder_mapping[label] + "/" + str(tweet_obj['id']) + ".txt"
                with open(folder, 'w') as save_file:
                    save_file.write(tweet_obj['text'])
            except KeyError:
                no_of_spam += 1
                continue
        print(no_of_spam)


def get_founta_tweet_ids():
    tweet_ids = []
    with open("data/founta/founta_only_ids.csv", "r+") as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            tweet_ids.append((int(row[0]), row[1]))
    return dict(tweet_ids)

def save_tweets_to_csv(tweet_ids):
    with open('data/data.json', 'r') as f, open('data/founta/founta_80k_spam.csv', 'a') as csv_file:
        data = json.load(f)
        writer = csv.writer(csv_file)
        #writer.writerow(["tweet", "label"])
        label_dict = {"hateful": 0, "abusive": 1, "normal": 2, "spam": 3}
        spam_count = 0
        for i, tweet_obj in enumerate(data):
            tweet = tweet_obj['text']
            label = tweet_ids[tweet_obj['id']]
            label = label_dict[label]
            if label == 3:
                spam_count += 1
            #writer.writerow([str(tweet), str(label)])
        print(spam_count)

save_tweets_to_csv(get_founta_tweet_ids())

def structure_davidson():
    with open("datasets_csv/davidson_25k.csv", "r+") as csvfile:
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            label = row[-2]
            folder = davidson_label_folder_mapping[label] + "/" + str(i) + ".txt"
            with open(folder, 'w') as save_file:
                save_file.write(row[-1])

def save_elsherief_txt():
    csv_files = glob.glob('data/elsherief/twitter_hashtag_based_datasets/*.csv')
    csv_files += glob.glob('data/elsherief/twitter_key_phrase_based_datasets/*.csv')
    tweet_ids = []
    for csv_path in csv_files:
        data = pd.read_csv(csv_path)
        tweet_ids += data.tweet_id.tolist()
    tweet_ids = list(set(tweet_ids))

    tc = TweetCollector()
    tweets = tc.lookup_tweets(tweet_ids)
    with open('data/elsherief/data.txt', 'a') as data_file:
        for tweet in tweets:
            data_file.write(tweet._json['text'].replace("\n", " ") + "\n\n")
    print("Of " + str(len(tweet_ids)) + " tweets, " + str(len(tweets)) + " tweets existed.")


def save_gao_huang_text():
    with open('data/gao_huang/fox-news-comments.json') as f, open('data/gao_huang/data.txt', 'a') as data_file:
        comments = [json.loads(line) for line in f]
        for comment in comments:
            data_file.write(comment['text'].replace("\n", " ") + "\n\n")


def save_gibert_text():
    paths = glob.glob('data/gibert/all_files/*.txt')
    ordrer_path = paths.sort(key=_natural_keys)
    sents_to_write = OrderedDict()
    for path in paths:
        with open(path, 'r') as f:
            doc_id = re.search('data/gibert/all_files/(.*)_', path).group(1)
            sent = f.readline()
            if 'nazi punk' in sent:
                print(sent, doc_id)
            if doc_id in sents_to_write:
                sents_to_write[doc_id].append(sent + "\n")
            else:
                sents_to_write[doc_id] = [sent + "\n"]
    with open('data/gibert/data.txt', 'a') as data_file:
        for doc_id, sents in sents_to_write.items():
            data_file.writelines(sents)
            data_file.write("\n")


def _atoi(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


def save_golbeck_text():
    with open("data/golbeck/onlineHarassmentDataset.tdf", "r+", encoding='latin1') as tdffile, open('data/golbeck/data.txt', 'a') as data_file:
        tdffile.readline()
        reader = csv.reader(tdffile, delimiter='\t')
        for row in reader:
            data_file.write(row[2] + "\n\n")


def save_waseem_hovy_text():
    with open('data/waseem_hovy/data.txt', 'a') as data_file:
        tweet_ids = []
        wh_data = pd.read_csv("data/waseem_hovy/NAACL_SRW_2016.csv")
        tweet_ids += wh_data.tweet_id.tolist()

        with open("data/waseem_hovy/NLP+CSS_2016.csv") as w_data:
            for row in w_data:
                tweet_ids.append(int(row.split()[0]))

        tc = TweetCollector()
        tweets = tc.lookup_tweets(list(set(tweet_ids)))
        with open('data/waseem_hovy/data.txt', 'a') as data_file:
            for tweet in tweets:
                data_file.write(tweet._json['text'].replace("\n", " ") + "\n\n")
        print("Of " + str(len(tweet_ids)) + " tweets, " + str(len(tweets)) + " tweets existed.")


def tokenize_tweets(data_path):
    with open(data_path, 'r') as data_file:
        all_sents = {}
        if data_path == "data/gibert/data.txt":
            gibert_sents = []
            for i, row in enumerate(data_file):
                if row.strip():
                    gibert_sents.append(row.strip("\n"))
                else:
                    if len(gibert_sents) > 1:
                        all_sents[i] = gibert_sents
                    else:
                        all_sents[i] = _split_single_sent(gibert_sents[0])
                    gibert_sents = []
        else:
            x = 0
            for i, row in enumerate(data_file):
                x += 1
                if row.strip():
                    row = _filter_tweet(row)
                    sents = sent_tokenize(row)
                    sents = [sent for sent in sents if len(sent) > 1]
                    if len(sents) == 1:
                        if len(sents[0].split()) > 2:
                            sents = _split_single_sent(sents[0])
                        else:
                            continue
                    all_sents[i] = sents
            print(x)
        return all_sents


def _filter_tweet(tweet):
    tknz_tweet = [word for word in tweet.split() if word != "RT" and word[0] != "@"]
    return ' '.join(tknz_tweet)


def _split_single_sent(sent):
    sent = sent.split()
    first_length = round(len(sent) / 2)
    first_sent = sent[0:first_length]
    second_sent = sent[first_length:]
    return [' '.join(first_sent), ' '.join(second_sent)]


def save_all_text():
    datapaths = [
        "data/elsherief/data.txt",
        "data/gao_huang/data.txt",
        "data/gibert/data.txt",
        "data/golbeck/data.txt",
        "data/waseem_hovy/data.txt"
    ]
    all_text = []
    for datapath in datapaths:
        all_text.append(tokenize_tweets(datapath))
    with open("data/all_all_data.txt", 'w+') as data_file:
        for docs in all_text:
            for _, text in docs.items():
                for sent in text:
                    data_file.write(sent + "\n")
                data_file.write("\n")


def check_overlap():
    six_data = pd.read_csv("misc/Toxic-Language-Detection-in-Online-Content-master/data/final/train.csv", encoding="latin1")
    twe_data = pd.read_csv("misc/Toxic-Language-Detection-in-Online-Content-master/data/final/test.csv")
    six_data_list = six_data.text.tolist()
    twe_data_list = twe_data.text.tolist()

    print(len(six_data_list))
    print(len(twe_data_list))

    six_data_set = set(six_data_list)
    twe_data_set = set(twe_data_list)

    print(len(six_data_set))
    print(len(twe_data_set))
    i = 0
    for tweet in twe_data_set:
        if tweet in six_data_set:
            i += 1
    print(i)
    print(i/len(twe_data_set))

def check_data():
    with open('data/all_all_data.txt') as f:
        two_lines = 0
        switch = False
        for i, row in enumerate(f):
            if not row.strip() and not switch:
                switch = True
            elif not row.strip() and switch:
                two_lines += 1
            else:
                switch = False
                two_lines = 0
            if two_lines:
                print(two_lines, i)

def check_founta():
    num = {'abusive': 0, 'normal': 0, 'spam': 0, 'hateful': 0}
    data = pd.read_csv('data/founta/founta_only_ids.csv')
    for label in data.label:
        num[label] += 1
    print(num)


