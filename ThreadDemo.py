import threading as th ;
import multiprocessing as mp;
def job(a,b):
    print("aaa")

#td.Thread 创建线程
#target 目标函数 注意目标函数只需要写名字，不需要写括号
#args=(a,b)需要传递的参数
t1 =th.Thread(target=job,args=(1,2))

#mp.Process 创建进程
#target 目标函数 注意目标函数只需要写名字，不需要写括号
#args=(a,b)需要传递的参数
p1 = mp.Process(target=job,args=(1,2))

#开启线程
t1.start();
#开启进程
mp.start();


