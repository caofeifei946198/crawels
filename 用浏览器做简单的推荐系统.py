import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#准备数据集，假设有五种类型的酒店
generatorNum = 5
hotelNum = 100
customerNum = 100000

generators = np.random.randint(5,size=(customerNum,generatorNum))
#用生成器随机生成数据
hotelComp = np.random.random(size=(generatorNum,hotelNum)) - 0.5
hotelRating = pd.DataFrame(generators.dot(hotelComp),index=["c%.6d"%i for i in range(100000)],
                           columns=["hotel_%.3d"%j for j in range(100)]).astype(int)

def normalize(s):#定义一个标准化的函数
    if s.std()>1e-6:
        return (s-s.mean())*s.std()**(-1)
    else:
        return (s-s.mean())
hotelRating_norm = hotelRating.apply(normalize)
hotelRating_norm_corr = hotelRating_norm.cov()

u,s,v = np.linalg.svd(hotelRating_norm_corr)
plt.plot(s,"o")
plt.title("singular value spectrum")
plt.show()
u_short = u[:,:5]
v_short = v[:5,:]
s_short = s[:5]

hotelRating_norm_corr_rebuild = pd.DataFrame(u_short.dot(np.diag(
    s_short).dot(v_short)),index=hotelRating_norm_corr.index,
    columns=hotelRating_norm_corr.keys())

#get the top components
top_components = hotelRating_norm.dot(u_short).dot(np.diag(np.power(s_short,-0.5)))
#classfication of each hotel
hotel_ind = 30
rating = hotelRating_norm.loc[:,"hotel_%.3d"%hotel_ind]
print("classification of the %dth hotel"%hotel_ind)
top_components.T.dot(rating)/customerNum


