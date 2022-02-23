import imageio
import jieba
import  wordcloud

#载入图片
from wordcloud import ImageColorGenerator

img=imageio.imread("66.png")


# 初始化词云
#括号里面的是生成图片的一些参数
#font_path='msyh.ttc' 是字体存放路径，msyh.ttc就是微软雅黑
#mask就是我们要生成的形状
#scale=15，scale的值越大图像密度越大越清晰
# stopwords={'的','了','说','是'} 屏蔽某些字
#contour_width=1, 轮廓的线宽
# contour_color='steelblue'  轮廓的线的颜色
w =wordcloud.WordCloud(width=1000,height=700,background_color='white',font_path='msyh.ttc',mask=img,scale=1,contour_width=1,contour_color='steelblue',stopwords={'的','了','说','是'})


#使用词云中的ImageColorGenerator方法 提取图片中的颜色
image_color=ImageColorGenerator(img)

#打开txt文件
txt=open("testDemo.txt",encoding="utf-8")
#读取text里面的文字
txtContent=txt.read();
#使用jieba的lcut函数来分割txt里面的文字但是分割出来的数据还不能直接使用，因为是个数组我们把他变成字符串就能使用了
txtContent=jieba.lcut(txtContent)
#join的用法

# str = "-";
# seq = ("a", "b", "c"); # 字符串序列
# print str.join( seq );

#a-b-c

txtContent=" ".join(txtContent)

# 词云生成图片所用的文字（他会删除掉一些没有什么含义的话比如am i...）
w.generate(txtContent)

#给词云图片中的对象重新上色
w.recolor(color_func=image_color)

#生成文件
w.to_file("test.png")
