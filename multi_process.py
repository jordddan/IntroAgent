import time
import vthread

@vthread.pool(10) # 只用加这一行就能实现6条线程池的包装
def foolfunc(res,num):
    time.sleep(1)
    print(f"foolstring, test2 foolnumb: {num}")
    res.append(num)

res = []

for i in range(10):
    foolfunc(res,i) # 加入装饰器后，这个函数变成往伺服线程队列里塞原函数的函数了
vthread.pool.wait()     
print(res)
# 不加装饰就是普通的单线程
# 只用加一行就能不破坏原来的代码结构直接实现线程池操作，能进行参数传递
