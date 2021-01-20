## 使用numba
import urllib2

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


urls = ['http://www.python.org', ..., 'http://baidu.com']
pool = pool = ThreadPool(4)
results = pool.map(urllib2.urlopen, urls]
pool.close()
pool.join()



## 使用concurrent.futures
import glob
import os
import cv2
import concurrent.futures


def load_and_resize(image_filename):
    img = cv2.imread(image_filename)
    img = cv2.resize(img, (600, 600))
    
with concurrent.futures.ProcessPoolExecutor() as executor:
    image_files = glob.glob("*.jpg")
    executor.map(load_and_resize, image_files)



## 使用multiprocessing
import multiprocessing
import time
import os
print("温馨提示：本机为",os.cpu_count(),"核CPU")  
 
def func(msg):
    print ("msg:", msg)
    time.sleep(3)
    print ("end")
 
if __name__ == "__main__":
    #这里开启了4个进程
    pool = multiprocessing.Pool(processes = 4)
    for i in xrange(4):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))   
 
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print ("Successfully")



## 使用Parallel
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
 
 
df = pd.read_csv(inputfile)
df = df.groupby(df.index)
retlist = Parallel(n_jobs=jobNum)(delayed(youFunc)(group, Funcarg1, Funcarg12) for name, group in tqdm(df))
df = pd.concat(retlist)

def Transanimal(animal):
        if animal=='dog':
            return A(animal)
        if animal=='pig':
            return B(animal)




