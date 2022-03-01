# pandas是用来导入、整理、清洗表格数据的专用工具，类似excel，但功能更加强大，导入的时候给pandas起个小名叫pd
import pandas as pd

# 用pandas的read_csv函数读取训练数据及测试数据，数据文件是.tsv格式的，也就是说数据用制表符\t分隔，类似于.csv文件的数据用逗号分隔
#也就说如果打开的是csv 后面的sep就是逗号
#你可能会问啥是tsv 简单来说是数据集，有打了标签，有的没打，这次的任务就是学了打标签的，然后标记没打标签的
data_train = pd.read_csv('./train.tsv',sep='\t')
data_test = pd.read_csv('./test.tsv',sep='\t')

#打印出训练集的个数
#data_train.shape= (156060, 4) 156060 列 4行，为什么比训练集多1？因为第四行就是标签
print(data_train.shape)
#打印出测试集的个数
#(data_test.shape) = (66292, 3) 66292 列 3行
print(data_test.shape)

#下面一些列代码是构建语料库
# 提取训练集中的文本内容
#将data_train里面的Phrase键名的值赋值给train_sentences
#也就是把文本内容拿出来
train_sentences = data_train['Phrase']

# 提取测试集中的文本内容
test_sentences = data_test['Phrase']

# sentences通过pandas的concat函数将训练集和测试集的文本内容合并到一起
sentences = pd.concat([train_sentences,test_sentences])



#打印出测试集+训练集文本的的总和个数
#(222352,) 共计222352个
print(sentences.shape)


# 提取训练集中的情感标签，一共是156060个标签
#将data_train里面的Sentiment键名的值赋值给label
label = data_train['Sentiment']

#打印出label.shape
print(label.shape)


#导入停词库，停词库中的词是一些废话单词和语气词，对情感分析没什么帮助
#splitlines()就是去除文本中的换行符 \n \t

#>>> 'ab c\n\nde fg\rkl\r\n'.splitlines()
# ['ab c', '', 'de fg', 'kl']
#为什么要导入停词库，因为英语语句中有一些没有什么意义的词
stop_words = open('./stop_words.txt',encoding='utf-8').read().splitlines()



#下面的代码使用词袋模型进行文本特征工程

# 用sklearn库中的CountVectorizer构建词袋模型
# analyzer='word'指的是以词为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
#好比一句话'I like you'
# 如果ngram_range = (2, 2)表示只选取前后的两个词构造词组合 :词向量组合为：’I like‘ 和 ’like you‘
# 如果ngram_range = (1, 3) 表示选取1到3个词做为组合方式: 词向量组合为: 'I', 'like', 'you', 'I like', 'like you', 'I like you' 构成词频标签

# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

from sklearn.feature_extraction.text import CountVectorizer
co = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    stop_words=stop_words,
    max_features=150000
)

# 使用语料库，构建词袋模型
#将sentences放入我们构建的词袋模型
#sentences通过pandas的concat函数将训练集和测试集的文本内容合并到一起
co.fit(sentences)

# 将训练集随机拆分为新的训练集和验证集，默认3:1,然后进行词频统计
# 在机器学习中，训练集相当于课后习题，用于平时学习知识。验证集相当于模拟考试，用于检验学习成果。测试集相当于高考，用于最终Kaggle竞赛打分。
# 新的训练集和验证集都来自于最初的训练集，都是有标签的。

from sklearn.model_selection import train_test_split
#train_sentences是我们训练集的内容

# x_train 训练集数据 （相当于课后习题）
# x_test 验证集数据 （相当于模拟考试题）
# y_train 训练集标签 （相当于课后习题答案）
# y_test 验证集标签（相当于模拟考试题答案）
x_train,x_test,y_train,y_test = train_test_split(train_sentences,label,random_state=1234)

print(x_train[0])
print(y_train[0])


# co.transform 用上面构建的词袋模型，把训练集和验证集中的每一个词都进行特征工程，变成向量
# x_train 训练集数据 （相当于课后习题）
# x_test 验证集数据 （相当于模拟考试题）
x_train = co.transform(x_train)
x_test = co.transform(x_test)

#构建分类器算法，对词袋模型处理后的文本进行机器学习和数据挖掘

#逻辑回归分类器

from sklearn.linear_model import LogisticRegression
#使用sklearn里面的LogisticRegression 逻辑回归模型 （速度慢，精度搞）
lg1 = LogisticRegression()
#把训练集中的数据和标签都放进去
# x_train 训练集数据 （相当于课后习题）
# y_train 训练集标签 （相当于课后习题答案）
#把训练集中的数据和标签都放进去
lg1.fit(x_train,y_train)
# x_test 验证集数据 （相当于模拟考试题）
# y_test 验证集标签（相当于模拟考试题答案）
# 使用lg1.score(x_test,y_test)  使用score来得到测试的结果 测试的内容是

print('词袋方法进行文本特征工程，使用sklearn默认的逻辑回归分类器，验证集上的预测准确率:',lg1.score(x_test,y_test))

#多项式朴素贝叶斯分类器（速度快，精度低）
#引用朴素贝叶斯进行分类训练和预测
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:',classifier.score(x_test,y_test))


# 用sklearn库中的TfidfVectorizer构建TF-IDF模型
# analyzer='word'指的是以词为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

# TF-IDF模型是专门用来过滤掉烂大街的词的，所以不需要引入停用词stop_words

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    # stop_words=stop_words,
    max_features=150000
)
# 使用语料库，构建词袋模型
#将sentences放入我们构建的词袋模型
#sentences通过pandas的concat函数将训练集和测试集的文本内容合并到一起
tf.fit(sentences)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_sentences,label,random_state=1234)
x_train = tf.transform(x_train)
x_test = tf.transform(x_test)

#构建分类器算法，对TF-IDF模型处理后的文本进行机器学习和数据挖掘

#朴素贝叶斯分类器
#引用朴素贝叶斯进行分类训练和预测
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:',classifier.score(x_test,y_test))

#逻辑回归分类器
# sklearn默认的逻辑回归模型
# C：正则化系数，C越小，正则化效果越强
# dual：求解原问题的对偶问题
lg2 = LogisticRegression(C=3, dual=True)
lg2.fit(x_train,y_train)
print('TF-IDF方法进行文本特征工程，使用增加了两个参数的逻辑回归模型，验证集上的预测准确率:',lg2.score(x_test,y_test))

#对比两个预测准确率可以看出，在逻辑回归中增加C和dual这两个参数可以提高验证集上的预测准确率，但如果每次都手动修改就太麻烦了。我们可以用sklearn提供的强大的网格搜索功能进行超参数的批量试验。
# 搜索空间：C从1到9。对每一个C，都分别尝试dual为True和False的两种参数。
# 最后从所有参数中挑出能够使模型在验证集上预测准确率最高的。

#使用网格搜索找出正则系数和对偶问题的最优解
from sklearn.model_selection import GridSearchCV
param_grid = {'C':range(1,10),
             'dual':[True,False]
              }
lgGS = LogisticRegression()
grid = GridSearchCV(lgGS, param_grid=param_grid,cv=3,n_jobs=-1)
grid.fit(x_train,y_train)

#打印出最优解
print(grid.best_params_)

#使用最优参数构建分类器

lg_final = grid.best_estimator_
print('经过网格搜索，找到最优超参数组合对应的逻辑回归模型，在验证集上的预测准确率:',lg_final.score(x_test,y_test))

#对测试集的数据进行预测，提交Kaggle竞赛最终结果
# 使用TF-IDF对测试集中的文本进行特征工程
test_X = tf.transform(data_test['Phrase'])
# 对测试集中的文本，使用lg_final逻辑回归分类器进行预测
predictions = lg_final.predict(test_X)
print(predictions)
print(predictions.shape)

# 将预测结果加在测试集中
#’A’列的所有记录，可以写df.loc[:, ‘A’]
# :表示所有
data_test.loc[:,'Sentiment'] = predictions

#获取完预测完毕后的结果
final_data = data_test.loc[:,['PhraseId','Sentiment']]
