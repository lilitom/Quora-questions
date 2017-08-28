"""
Detecting duplicate quora questions
feature engineering

"""
# GuoHao
# 2017-7-21
# -*- coding：utf-8 -*-

import cPickle
import re
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
import distance
import string

SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x

def wmd(s1, s2): #word move distance
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2) 


def norm_wmd(s1, s2):#归一化的 wmd
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):#简答的w2v加和后归一化
    words = preprocess(s)
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())
	
def get_token_features(q1, q2):#总函数 提取特征
    token_features = [0.0]*10

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def get_weights(count, eps = 10, min_count = 2):
	if count < min_count:
		return 0
	else:
		return 1 / float(count**2 + eps)


def word_match_share(row): #匹配的比例
	q1w = []
	q2w = []
	# 去掉标点符号
	no_pun_q1 = regex.sub('', str(row['question1']))
	no_pun_q2 = regex.sub('', str(row['question2']))

	for word in no_pun_q1.lower().split():
		if word not in stops:
			q1w.append(word)

	for word in no_pun_q2.lower().split():
		if word not in stops:
			q2w.append(word)

	if len(q1w) * len(q2w) == 0:
		return 0
	else:
		return len(set(q1w) & set(q2w)) / float(len(set(q1w + q2w)) + 1)

def tfidf_word_match_share(row): #共享词的权重
	q1w = []
	q2w = []
	no_pun_q1 = regex.sub('', str(row['question1']))
	no_pun_q2 = regex.sub('', str(row['question2']))
	for word in no_pun_q1.lower().split():
		if word not in stops:
			q1w.append(word)
	for word in no_pun_q2.lower().split():
		if word not in stops:
			q2w.append(word)
	if len(q1w) * len(q2w) == 0:
		return 0
	else:
		share_weights = [weights.get(w,0) for w in list(set(q1w) & set(q2w))]
		union_weights = [weights.get(w,0) for w in list(set(q1w + q2w))]
		return np.sum(share_weights) / float(np.sum(union_weights) + 1)


print("token features...")
regex = re.compile('[%s]' % re.escape(string.punctuation))
stops = set(stopwords.words("english"))
print("generate bow and tfidf features...")
data =  pd.read_csv('..data/train.csv')
df_test = pd.read_csv('..data/test.csv')
train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist()
					 + df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
words = regex.sub("", (" ".join(train_qs))).lower().split()
counts = Counter(words)
weights = {word: get_weights(count) for word, count in counts.items()}


data = data.drop(['id', 'qid1', 'qid2'], axis=1)
data['word_match_share']=data.apply(word_match_share,axis=1,raw=True)
data['tfidf_word_match_share']=data.apply(tfidf_word_match_share,axis=1,raw=True)
data["question1"] = data["question1"].fillna("").apply(preprocess)
data["question2"] = data["question2"].fillna("").apply(preprocess)
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = data.len_q1 - data.len_q2
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


model = gensim.models.KeyedVectors.load_word2vec_format('..data/GoogleNews-vectors-negative300.bin.gz', binary=True)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
norm_model = gensim.models.KeyedVectors.load_word2vec_format('..data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)
question2_vectors  = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)


token_features = data.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
data["cwc_min"]       = list(map(lambda x: x[0], token_features))
data["cwc_max"]       = list(map(lambda x: x[1], token_features))
data["csc_min"]       = list(map(lambda x: x[2], token_features))
data["csc_max"]       = list(map(lambda x: x[3], token_features))
data["ctc_min"]       = list(map(lambda x: x[4], token_features))
data["ctc_max"]       = list(map(lambda x: x[5], token_features))
data["last_word_eq"]  = list(map(lambda x: x[6], token_features))
data["first_word_eq"] = list(map(lambda x: x[7], token_features))
data["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
data["mean_len"]      = list(map(lambda x: x[9], token_features))

data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

#保留特征
data.to_csv('..data/quora_features.csv', index=False)
