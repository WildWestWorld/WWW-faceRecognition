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

# 通过pandas的concat函数将训练集和测试集的文本内容合并到一起
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
stop_words = open('./stop_words.txt',encoding='utf-8').read().splitlines(



#下面的代码使用词袋模型进行文本特征工程

# 用sklearn库中的CountVectorizer构建词袋模型
# 词袋模型的详细介绍请看子豪兄的视频
# analyzer='word'指的是以词为单位进行分析，对于拉丁语系语言，有时需要以字母'character'为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

from sklearn.feature_extraction.text import CountVectorizer
co = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    stop_words=stop_words,
    max_features=150000
)