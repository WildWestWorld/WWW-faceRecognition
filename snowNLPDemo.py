import imageio
import jieba
import snownlp

#打开txt
import wordcloud
from wordcloud import ImageColorGenerator

txt= open("testDemo.txt",encoding="utf-8")
#获取txt的内容
txtContent = txt.read();
#使用jieba让划分每个词
txtSentiments=jieba.lcut(txtContent)

passiveWord=[]
negativeWord=[]

for item in txtSentiments:

    #使用snownlp中的SnowNLP方法 在方法里面放入要识别的文本 返回值代表着被识别后的文字
    word = snownlp.SnowNLP(item)
    #使用文字中的setiments属性就能知道该文字的情感色彩是积极还是消极了 该值在0-1之间，越大越积极
    feeling = word.sentiments
    if(feeling>0.9):
        #如果该词不在passiveWord列表内
        if(item not in passiveWord ):
            passiveWord.append(item);
    if(feeling<0.2):
        #如果该词不在passiveWord列表内
        if (item not in negativeWord ):
            negativeWord.append(item)

print(passiveWord)
print(negativeWord)

# str = "-";
# seq = ("a", "b", "c"); # 字符串序列
# print str.join( seq );

#a-b-c

#列表变成字符串
passiveStr=" ".join(passiveWord)
negativeStr=" ".join(negativeWord)

img=imageio.imread("66.png")


# 初始化词云
#括号里面的是生成图片的一些参数
#font_path='msyh.ttc' 是字体存放路径，msyh.ttc就是微软雅黑
#mask就是我们要生成的形状
#scale=15，scale的值越大图像密度越大越清晰
# stopwords={'的','了','说','是'} 屏蔽某些字
#contour_width=1, 轮廓的线宽
# contour_color='steelblue'  轮廓的线的颜色
w1 =wordcloud.WordCloud(width=1000,height=700,background_color='white',font_path='msyh.ttc',mask=img,scale=1,contour_width=1,contour_color='steelblue',stopwords={'的','了','说','是'})
w2 =wordcloud.WordCloud(width=1000,height=700,background_color='white',font_path='msyh.ttc',mask=img,scale=1,contour_width=1,contour_color='steelblue',stopwords={'的','了','说','是'})


#使用词云中的ImageColorGenerator方法 提取图片中的颜色
image_color=ImageColorGenerator(img)




# 词云生成图片所用的文字（他会删除掉一些没有什么含义的话比如am i...）
w1.generate(passiveStr)
w2.generate(negativeStr)

#给词云图片中的对象重新上色
w1.recolor(color_func=image_color)
w2.recolor(color_func=image_color)
#生成文件
w1.to_file("test1.png")
w2.to_file("test2.png")