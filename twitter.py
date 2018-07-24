# -*- coding: utf-8 -*-
###导入模组###
from textblob import TextBlob
import tweepy
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist
import re
from numpy import matrix, zeros
import got
import datetime
import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd
from pandas_datareader import data as dt
from matplotlib import style
import scipy.stats as stats
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error

###为捕捉器授权###
consumerKey = 'CnVgWZd7njUfV4rUmQAz823Vh'
consumerSecret = 'jlCEbO4a95MBFsDx1GPeLhTIYK0DQar7Ql0WDJASovSFbmfrYM'
accessToken = '974581663706460160-2OiECrAZkG9WttrMaI8bHAEBCxyDfhl'
accessTokenSecret = '0kXvtGZzrXAIgTgrUwJpxIwgoOsA2JLydLuLkAwNxBYPr'
auth = tweepy.OAuthHandler(consumer_key = consumerKey, consumer_secret = consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)


###计算标杆文本向量###
#专业组：白宫（@WhiteHouse）、华尔街日报（@WSJ）、彭博社（@business）
professional_list = ['WhiteHouse', 'WSJ', 'business']
features = matrix(zeros((3,7))) #文本特征向量
j = 0
for ProList in professional_list:    
    Status = api.user_timeline(screen_name = ProList, count = 200, exclude_replies = True)
    for stat in Status:
        PL_text = stat.text
        #去除无意义字符
        PL_text = re.sub('https://.*', '', PL_text)
        PL_text = re.sub('#', '', PL_text)
        PL_text = re.sub('@', '', PL_text)           
        features[j,1] += len(PL_text) #总字符数
        features[j,3] += len(FreqDist(word_tokenize(PL_text))) #词汇总数
        features[j,4] += len(sent_tokenize(PL_text)) #句子数量
        word_len = 0
        for word in word_tokenize(PL_text):
            word_len += len(word)
            if len(word)<5:
                features[j,5] += 1 #短词数量（字符数小于5的单词）
        i=0
        while i<=len(PL_text)-1:             
              if PL_text[i]>="A" and PL_text[i]<="Z":
                 features[j,0]+=1 #大写字母数
              elif PL_text[i]=='0' or PL_text[i]=='1'or PL_text[i]=='2'or PL_text[i]=='3'or PL_text[i]=='4'or PL_text[i]=='5'or PL_text[i]=='6'or PL_text[i]=='7'or PL_text[i]=='8'or PL_text[i]=='9':  
                  features[j,2] +=1 #数字字符数
              if(PL_text[i] == '?') or (PL_text[i] == '!'):
                  features[j,6] += 1 #语义型标点符号数量 ！？
              i += 1
    features[j,:] = features[j,:]/len(Status)
    j +=1
    
#非专业组：唐纳德川普（@realDonaldTrump）、泰勒斯威夫特（@taylorswift13）、科比布莱恩特（@kobebryant）
Nprofessional_list = ['realDonaldTrump', 'taylorswift13', 'kobebryant']
features = matrix(zeros((3,7))) #文本特征向量
j = 0
for NProList in Nprofessional_list:    
    Status = api.user_timeline(screen_name = NProList, count = 200, exclude_replies = True)
    for stat in Status:
        PL_text = stat.text
        PL_text = re.sub('https://.*', '', PL_text)
        PL_text = re.sub('#', '', PL_text)
        PL_text = re.sub('@', '', PL_text)           
        features[j,1] += len(PL_text) #总字符数
        features[j,3] += len(FreqDist(word_tokenize(PL_text))) #词汇总数
        features[j,4] += len(sent_tokenize(PL_text)) #句子数量
        word_len = 0
        for word in word_tokenize(PL_text):
            word_len += len(word)
            if len(word)<5:
                features[j,5] += 1 #短词数量（字符数小于5的单词）
        i=0
        while i<=len(PL_text)-1:             
              if PL_text[i]>="A" and PL_text[i]<="Z":
                 features[j,0]+=1 #大写字母数
              elif PL_text[i]=='0' or PL_text[i]=='1'or PL_text[i]=='2'or PL_text[i]=='3'or PL_text[i]=='4'or PL_text[i]=='5'or PL_text[i]=='6'or PL_text[i]=='7'or PL_text[i]=='8'or PL_text[i]=='9':  
                  features[j,2] +=1 #数字字符数
              if(PL_text[i] == '?') or (PL_text[i] == '!'):
                  features[j,6] += 1 #语义型标点符号数量 ！？
              i += 1
    features[j,:] = features[j,:]/len(Status)
    j +=1


###计算指定公司两年褒贬值###
#输入专业组、非专业组标杆向量
proArray = matrix([5.271,96.866,0.374,16.832,1.249,9.933,0.055])
nproArray = matrix([6.947,77.347,0.561,13.547,1.473,9.026,0.293])
research_list = 'walmart' #沃尔玛
vol = 5 #每日爬取推特100条
startA = '2016-11-22'#从2016/1/1至2017/12/31
startB = '2016-11-23'
end = '2017-01-01'
datestartA = datetime.datetime.strptime(startA, '%Y-%m-%d')
datestartB = datetime.datetime.strptime(startB, '%Y-%m-%d')
dateend = datetime.datetime.strptime(end, '%Y-%m-%d')    
Polarity = matrix(zeros((200,1)))
t = 0
while datestartA<dateend:   
    currentTimeA = datestartA.strftime('%Y-%m-%d')
    currentTimeB = datestartB.strftime('%Y-%m-%d')
    datestartA += datetime.timedelta(days=1)
    datestartB += datetime.timedelta(days=1)
    print currentTimeA, currentTimeB  
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(research_list).setSince(currentTimeA).setUntil(currentTimeB).setMaxTweets(vol)    
    for k in range(vol):
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)[k].text
        Polarity_init = TextBlob(tweet).sentiment.polarity
        print Polarity_init, k
        #计算该文本的文本特征向量
        features_indiv = matrix(zeros((1,7)))
        tweet = re.sub('http.*', '', tweet)
        tweet = re.sub('#', '', tweet)
        tweet = re.sub('@', '', tweet) 
        features_indiv[0,1] += len(tweet) 
        features_indiv[0,3] += len(FreqDist(word_tokenize(tweet)))
        features_indiv[0,4] += len(sent_tokenize(tweet))
        word_len = 0
        for word in word_tokenize(tweet):
            word_len += len(word)
            if len(word)<5:
                features_indiv[0,5] += 1 
        i=0
        while i<=len(tweet)-1:             
              if tweet[i]>="A" and tweet[i]<="Z":
                 features_indiv[0,0]+=1 
              elif tweet[i]=='0' or tweet[i]=='1'or tweet[i]=='2'or tweet[i]=='3'or tweet[i]=='4'or tweet[i]=='5'or tweet[i]=='6'or tweet[i]=='7'or tweet[i]=='8'or tweet[i]=='9':  
                  features_indiv[0,2] +=1 
              elif(tweet[i] == '?') or (tweet[i] == '!'):
                  features_indiv[0,6] += 1
              i += 1
        #计算余弦距离确定该文本所属分组
        dist1 = pdist(np.vstack([features_indiv,proArray]),'cosine')
        dist2 = pdist(np.vstack([features_indiv,nproArray]),'cosine')
        if dist1>dist2:
            Polarity[t,0] += Polarity_init*1.3
        else:
            Polarity[t,0] += Polarity_init    
        k += 1
    Polarity[t,0] = Polarity[t,0]/vol
    t += 1        
  
    
##获取公司股票信息并绘制相关折线图
#获取沃尔玛股票信息
WMT = dt.get_data_google('wmt', start='2015-12-31', end='2018-01-01')
print WMT.head()
WMT.to_csv('walmart.csv')
#绘制折线图
style.use('ggplot')
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
df = pd.read_csv('walmart.csv', index_col='Date', parse_dates=True)
df['H-L'] = df.High - df.Low
ma = pd.rolling_mean(df.Close, 10)
#股价相关信息折线图
ax1 = plt
ax1.plot(df.Close, label='Walmart')
ax1.plot(ma, label='10MA')
plt.legend()
ax1.xlabel('date')
ax1.ylabel('price')
ax1.title('Price line chart for Walmart')
plt.show()
#高低价差条形图
ax2 = plt
ax2.plot(df['H-L'], label='H-L')
plt.legend()
ax1.xlabel('date')
ax1.ylabel('difference in price')
ax1.title('High and low price spread bar chart')
plt.show()
#交易量折线图
ax3 = plt
ax3.plot(df['Volume'], label='Volume')
plt.legend()
ax1.xlabel('date')
ax1.ylabel('Volume')
ax1.title('Trading volume line chart')
plt.show()
#褒贬值折线图
ax4 = plt
ax4.plot(df_EXP1['A'], label='sentiment value')
plt.legend()
ax4.xlabel('date')
ax4.ylabel('value')
ax4.title('Sentiment value line chart')
plt.show()


###利用SVM拟合模型并计算评价指标###
#计算序列的相关性，并做显著性检验
df_EXP1 = pd.read_csv('EXP1.csv', index_col='Date', parse_dates=True)
r = matrix(zeros((3,3)))
p =matrix(zeros((3,3)))
r[0,0],p[0,0] =stats.pearsonr(df_EXP1['A'],df_EXP1['Rt']) #皮尔逊相关系数、p值
r[0,1],p[0,1] =stats.pearsonr(df_EXP1['B'],df_EXP1['Rt']) #横轴为情感项，纵轴为金融变量
r[0,2],p[0,2] =stats.pearsonr(df_EXP1['C'],df_EXP1['Rt'])
r[1,0],p[1,0] =stats.pearsonr(df_EXP1['A'],df_EXP1['VVt'])
r[1,1],p[1,1] =stats.pearsonr(df_EXP1['B'],df_EXP1['VVt'])
r[1,2],p[1,2] =stats.pearsonr(df_EXP1['C'],df_EXP1['VVt'])
r[2,0],p[2,0] =stats.pearsonr(df_EXP1['A'],df_EXP1['PVt'])
r[2,1],p[2,1] =stats.pearsonr(df_EXP1['B'],df_EXP1['PVt'])
r[2,2],p[2,2] =stats.pearsonr(df_EXP1['C'],df_EXP1['PVt'])

##SVM模型标准误减小值##
df_EXP1 = pd.read_csv('EXP1.csv', index_col='Date', parse_dates=True)
svr = svm.SVR()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.25,0.5,1,2,4,8,16]}
clf = GridSearchCV(svr, parameters, cv=3)
#不含情感项的模型#
#振幅
y11 = df_EXP1['Rt']
x11 = df_EXP1['Rt1'].reshape(-1,1)
x11_train, x11_test, y11_train, y11_test = train_test_split(x11, y11, random_state=1, train_size=0.8)
clf.fit(x11_train, y11_train)
print mean_absolute_error(y11_test, clf.predict(x11_test))

#交易量
y12 = df_EXP1['VVt']
x12 = df_EXP1['VVt1'].reshape(-1,1)
x12_train, x12_test, y12_train, y12_test = train_test_split(x12, y12, random_state=1, train_size=0.8)
clf.fit(x12_train, y12_train)
print mean_absolute_error(y12_test, clf.predict(x12_test))

#隔日涨跌率
y13 = df_EXP1['PVt']
x13 = df_EXP1['PVt1'].reshape(-1,1)
x13_train, x13_test, y13_train, y13_test = train_test_split(x13, y13, random_state=1, train_size=0.8)
clf.fit(x13_train, y13_train)
print mean_absolute_error(y13_test, clf.predict(x13_test))

#含情感项的模型#
#振幅
y14 = df_EXP1['Rt']
x14 = pd.concat([df_EXP1['A'], df_EXP1['B'], df_EXP1['Rt1']], axis=1)
x14_train, x14_test, y14_train, y14_test = train_test_split(x14, y14, random_state=1, train_size=0.8)
clf.fit(x14_train, y14_train)
print mean_absolute_error(y14_test, clf.predict(x14_test))

#交易量
y15 = df_EXP1['VVt']
x15 = pd.concat([df_EXP1['A'], df_EXP1['B'], df_EXP1['VVt1']], axis=1)
x15_train, x15_test, y15_train, y15_test = train_test_split(x15, y15, random_state=1, train_size=0.8)
clf.fit(x15_train, y15_train)
print mean_absolute_error(y15_test, clf.predict(x15_test))

#隔日涨跌率
y16 = df_EXP1['PVt']
x16 = pd.concat([df_EXP1['A'], df_EXP1['B'], df_EXP1['PVt1']], axis=1)
x16_train, x16_test, y16_train, y16_test = train_test_split(x16, y16, random_state=1, train_size=0.8)
clf.fit(x16_train, y16_train)
print mean_absolute_error(y16_test, clf.predict(x16_test))

##趋势预测准确度##
df_EXP3 = pd.read_csv('EXP3.csv', index_col='Date', parse_dates=True)
svc = svm.SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.25,0.5,1,2,4,8,16]}
clf = GridSearchCV(svc, parameters, cv=10)
#clf = svm.SVC(kernel='rbf', gamma=20, decision_function_shape='ovr')
#不含情感项的模型#
#振幅
y31 = df_EXP3['DRt']
x31 = df_EXP3['DRt1'].reshape(-1,1)
x31_train, x31_test, y31_train, y31_test = train_test_split(x31, y31, random_state=1, train_size=0.8)
clf.fit(x31_train, y31_train)
print(classification_report(y31_test,clf.predict(x31_test)))  

#交易量
y32 = df_EXP3['DVVt']
x32 = df_EXP3['DVVt1'].reshape(-1,1)
x32_train, x32_test, y32_train, y32_test = train_test_split(x32, y32, random_state=1, train_size=0.8)
clf.fit(x32_train, y32_train)
print(classification_report(y32_test,clf.predict(x32_test)))

#隔日涨跌率
y33 = df_EXP3['DPVt']
x33 = df_EXP3['DPVt1'].reshape(-1,1)
x33_train, x33_test, y33_train, y33_test = train_test_split(x33, y33, random_state=1, train_size=0.8)
clf.fit(x33_train, y33_train)
print(classification_report(y33_test,clf.predict(x33_test)))   

#含情感项的模型#
#振幅
y34 = df_EXP3['DRt']
x34 = pd.concat([df_EXP3['DA'], df_EXP3['DB'], df_EXP3['DRt1']], axis=1)
x34_train, x34_test, y34_train, y34_test = train_test_split(x34, y34, random_state=1, train_size=0.8)
clf.fit(x34_train, y34_train)
print(classification_report(y34_test,clf.predict(x34_test))) 
    
#交易量    
y35 = df_EXP3['DVVt']
x35 = pd.concat([df_EXP3['DA'], df_EXP3['DB'], df_EXP3['DRt1']], axis=1)
x35_train, x35_test, y35_train, y35_test = train_test_split(x35, y35, random_state=1, train_size=0.8)
clf.fit(x35_train, y35_train)
print(classification_report(y35_test,clf.predict(x35_test))) 
  
#隔日涨跌率   
y36 = df_EXP3['DPVt']
x36 = pd.concat([df_EXP3['DA'], df_EXP3['DB'], df_EXP3['DPVt1']], axis=1)
x36_train, x36_test, y36_train, y36_test = train_test_split(x36, y36, random_state=1, train_size=0.8)
clf.fit(x36_train, y36_train)
print(classification_report(y36_test,clf.predict(x36_test)))
   
    
    
    