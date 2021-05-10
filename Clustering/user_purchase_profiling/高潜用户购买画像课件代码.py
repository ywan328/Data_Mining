#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
# 忽略以为版本信息出现的警告
import warnings
warnings.filterwarnings('ignore') 


# In[ ]:


# 检测文件来源
def user_action_check():
    # 读入数据
    df_user = pd.read_csv('data/Data_User.csv', encoding='gbk') 
    # series将数组转换为DataFrame格式
    df_sku = df_user.loc[:, 'user_id'].to_frame()  

    df_month2 = pd.read_csv('data/Data_Action_201602.csv', encoding='gbk')
    print('Is action of Feb. from User file? ',
          len(df_month2) == len(pd.merge(df_sku, df_month2)))

    df_month3 = pd.read_csv('data/Data_Action_201603.csv', encoding='gbk')
    print('Is action of Mar. from User file? ',
          len(df_month3) == len(pd.merge(df_sku, df_month3)))
    df_month4 = pd.read_csv('data/Data_Action_201604.csv', encoding='gbk')
    print('Is action of Apr. from User file? ',
          len(df_month4) == len(pd.merge(df_sku, df_month4)))


user_action_check()

# 2、3、4月份的数据是否来自User文件


# In[ ]:


def deduplicate(filepath, filename, newpath):
    # 读入数据
    df_file = pd.read_csv(filepath, encoding='gbk')  
    # 样本的行号/长度
    before = df_file.shape[0]  
    # 去重复值
    df_file.drop_duplicates(inplace=True)  
    # 再查看有多少样本数/长度
    after = df_file.shape[0]
    # 前后样本数的差值
    n_dup = before - after  
    print('No. of duplicate records for ' + filename + ' is: ' + str(n_dup))
    if n_dup != 0:
        df_file.to_csv(newpath, index=None)
    else:
        print('no duplicate records in ' + filename)


# In[ ]:


# 检测重复值个数，并生成新文件
# deduplicate('data/Data_Action_201602.csv', 'Feb. action', 'data/Data_Action_201602_dedup.csv')
deduplicate('data/Data_Action_201603.csv', 'Mar. action',
            'data/Data_Action_201603_dedup.csv')
deduplicate('data/Data_Action_201604.csv', 'Apr. action',
            'data/Data_Action_201604_dedup.csv')
deduplicate('data/Data_Comment.csv', 'Comment', 'data/Data_Comment_dedup.csv')
deduplicate('data/Data_Product.csv', 'Product', 'data/Data_Product_dedup.csv')
deduplicate('data/Data_User.csv', 'User', 'data/Data_User_dedup.csv')

# 第一行重复数据有7085038，说明同一个商品买了多个
# 第二行重复数据有3672710
# 第三行重复数据为0


# In[ ]:


# 检测重复值引起的原因
df_month2 = pd.read_csv('data/Data_Action_201602.csv', encoding='gbk')
# 检查重复值
IsDuplicated = df_month2.duplicated()
df_d = df_month2[IsDuplicated]
# 发现重复数据大多数都是由于浏览（1），或者点击(6)产生
df_d.groupby('type').count() 


# In[ ]:


# 由于注册时间是系统错误造成，如果行为数据中没有在4月15号之后的数据的话，
# 那么说明这些用户还是正常用户，并不需要删除。
import pandas as pd
df_user = pd.read_csv('data/Data_User.csv', encoding='gbk')
# 找到用户注册时间这一列
df_user['user_reg_tm'] = pd.to_datetime(df_user['user_reg_tm'])  
df_user.loc[df_user.user_reg_tm >= '2016-4-15']


# In[ ]:


# 查看异常操作记录
df_month = pd.read_csv('data/Data_Action_201604.csv')
df_month['time'] = pd.to_datetime(df_month['time'])
df_month.loc[df_month.time >= '2016-4-16']
# 结论：说明用户没有异常操作数据，所以这一批用户不删除


# In[ ]:


# 数据类型转换
import pandas as pd
df_month = pd.read_csv('data/Data_Action_201602.csv', encoding='gbk')
df_month['user_id'] = df_month['user_id'].apply(lambda x: int(x))
print(df_month['user_id'].dtype)
df_month.to_csv('data/Data_Action_201602.csv', index=None)
df_month = pd.read_csv('data/Data_Action_201603.csv', encoding='gbk')
df_month['user_id'] = df_month['user_id'].apply(lambda x: int(x))
print(df_month['user_id'].dtype)
df_month.to_csv('data/Data_Action_201603.csv', index=None)
df_month = pd.read_csv('data/Data_Action_201604.csv', encoding='gbk')
df_month['user_id'] = df_month['user_id'].apply(lambda x: int(x))
print(df_month['user_id'].dtype)
df_month.to_csv('data/Data_Action_201604.csv', index=None)


# In[ ]:


import pandas as pd
df_user = pd.read_csv('data/Data_User.csv', encoding='gbk')


def tranAge(x):
    if x == u'15岁以下':
        x = '1'
    elif x == u'16-25岁':
        x = '2'
    elif x == u'26-35岁':
        x = '3'
    elif x == u'36-45岁':
        x = '4'
    elif x == u'46-55岁':
        x = '5'
    elif x == u'56岁以上':
        x = '6'
    return x


df_user['age'] = df_user['age'].apply(tranAge)
# 有14412个没有透露年龄，在年龄值为3时候最多，属于”26—35岁“
print(df_user.groupby(
    df_user['age']).count())  
df_user.to_csv('data/Data_User.csv', index=None)


# In[ ]:


#重定义文件名
ACTION_201602_FILE = "data/Data_Action_201602.csv"
ACTION_201603_FILE = "data/Data_Action_201603.csv"
ACTION_201604_FILE = "data/Data_Action_201604.csv"
COMMENT_FILE = "data/Data_Comment.csv"
PRODUCT_FILE = "data/Data_Product.csv"
USER_FILE = "data/Data_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"


# In[ ]:


# 导入相关包
import pandas as pd
import numpy as np
from collections import Counter


# In[ ]:


# 功能函数: 对每一个user分组的数据进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    # 统计用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[[
        'user_id', 'browse_num', 'addcart_num', 'delcart_num', 'buy_num',
        'favor_num', 'click_num'
    ]]


# In[ ]:


# 对action数据进行统计
# 因为由于用户行为数据量较大,一次性读入可能造成内存错误(Memory Error)
# 因而使用pandas的分块(chunk)读取.根据自己调节chunk_size大小


def get_from_action_data(fname, chunk_size=50000):
    # iterator:返回一个可迭代的对象
    reader = pd.read_csv(fname, header=0, iterator=True, encoding='gbk')
    chunks = []
    loop = True
    while loop:
        try:
            # 只读取user_id和type两个字段
            chunk = reader.get_chunk(chunk_size)[["user_id", "type"]]
            chunks.append(chunk)
        # 迭代器迭代完成后会产生StopIteratio异常
        except StopIteration:  
            loop = False
            print("Iteration is stopped")
    # 将块拼接为pandas dataframe格式
    df_ac = pd.concat(chunks, ignore_index=True)
    # 按user_id分组，对每一组进行统计，as_index 表示无索引形式返回数据
    df_ac = df_ac.groupby(['user_id'], as_index=False).apply(add_type_count)
    # 将重复的行丢弃
    df_ac = df_ac.drop_duplicates('user_id')

    return df_ac


# In[ ]:


# 将各个action数据的统计量进行聚合
def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))  
    # 对上面三个表进行拼接
    df_ac = pd.concat(df_ac, ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['user_id'], as_index=False).sum()
    #　构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac[
        'addcart_num']  # 加了多少次购物车才买，购买率
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac[
        'browse_num']  # 浏览了多少次才买
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac[
        'click_num']  # 点击了多少次才买
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac[
        'favor_num']  # 喜欢了多少个才买

    # 将大于１的转化率字段置为１(100%)，确保数据没有问题
    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac


# In[ ]:


#　从Data_User表中抽取需要的字段
def get_from_jdata_user():
    df_usr = pd.read_csv(USER_FILE, header=0)
    df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    return df_usr


# In[ ]:


# 执行目的是得到大表
# 统计数据
user_base = get_from_jdata_user()
# 数据聚合
user_behavior = merge_action_data()


# In[ ]:


# 连接成一张表，类似于SQL的左连接(left join)
user_behavior = pd.merge(user_base, user_behavior, on=['user_id'], how='left')
# 保存中间结果为user_table.csv
user_behavior.to_csv(USER_TABLE_FILE, index=False)


# In[ ]:


user_table = pd.read_csv(USER_TABLE_FILE)
user_table.head()


# In[ ]:


#定义文件名
ACTION_201602_FILE = "data/Data_Action_201602.csv"
ACTION_201603_FILE = "data/Data_Action_201603.csv"
ACTION_201604_FILE = "data/Data_Action_201604.csv"
COMMENT_FILE = "data/Data_Comment.csv"
PRODUCT_FILE = "data/Data_Product.csv"
USER_FILE = "data/Data_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"


# In[ ]:


# 导入相关包
import pandas as pd
import numpy as np
from collections import Counter


# In[ ]:


# 读取Product中商品
def get_from_jdata_product():
    df_item = pd.read_csv(PRODUCT_FILE, header=0,encoding='gbk')
    return df_item


# In[ ]:


# 对每一个商品分组进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['sku_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]


# In[ ]:


#对action中的数据进行统计
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[["sku_id", "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)

    df_ac = df_ac.groupby(['sku_id'], as_index=False).apply(add_type_count)
    # Select unique row
    df_ac = df_ac.drop_duplicates('sku_id')

    return df_ac


# In[ ]:


# 获取评论中的商品数据,如果存在某一个商品有两个日期的评论，我们取最晚的那一个
def get_from_jdata_comment():
    df_cmt = pd.read_csv(COMMENT_FILE, header=0)
    df_cmt['dt'] = pd.to_datetime(df_cmt['dt'])
    # 找到最晚评论的索引
    idx = df_cmt.groupby(['sku_id'])['dt'].transform(max) == df_cmt['dt']
    df_cmt = df_cmt[idx]
    
    # 返回，商品ID，评论数，是否有差评，差评率
    return df_cmt[['sku_id', 'comment_num',
                   'has_bad_comment', 'bad_comment_rate']]


# In[ ]:


# 将各个action数据的统计量进行聚合
def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.groupby(['sku_id'], as_index=False).sum()

    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']

    df_ac.ix[df_ac['buy_addcart_ratio'] > 1., 'buy_addcart_ratio'] = 1.
    df_ac.ix[df_ac['buy_browse_ratio'] > 1., 'buy_browse_ratio'] = 1.
    df_ac.ix[df_ac['buy_click_ratio'] > 1., 'buy_click_ratio'] = 1.
    df_ac.ix[df_ac['buy_favor_ratio'] > 1., 'buy_favor_ratio'] = 1.

    return df_ac


# In[ ]:


item_base = get_from_jdata_product()
item_behavior = merge_action_data()
item_comment = get_from_jdata_comment()

# SQL: left join
item_behavior = pd.merge(item_base, item_behavior, on=['sku_id'], how='left')
item_behavior = pd.merge(item_behavior,
                         item_comment,
                         on=['sku_id'],
                         how='left')

item_behavior.to_csv(ITEM_TABLE_FILE, index=False)


# In[ ]:


item_table = pd.read_csv(ITEM_TABLE_FILE)
item_table.head()


# In[ ]:


import pandas as pd

df_user = pd.read_csv('data/User_table.csv', header=0)
pd.options.display.float_format = '{:,.3f}'.format  #输出格式设置，保留三位小数
df_user.describe()

# 第一行中根据User_id统计发现有105321个用户，发现有几个用户没有age,sex字段，
# 而且根据浏览、加购、删购、购买等记录却只有105180条记录，
# 说明存在用户无任何交互记录，因此可以删除上述用户。


# In[ ]:


# 选择性删除没有年龄的数据
# 如果有空值启动此行
# df_user.drop(df_user[df_user['age'].isnull()], axis=0, inplace=True)


# In[ ]:


# 删除无交互记录的用户
df_naction = df_user[(df_user['browse_num'].isnull())
                     & (df_user['addcart_num'].isnull()) &
                     (df_user['delcart_num'].isnull()) &
                     (df_user['buy_num'].isnull()) &
                     (df_user['favor_num'].isnull()) &
                     (df_user['click_num'].isnull())]
df_user.drop(df_naction.index, axis=0, inplace=True)
print(len(df_user))


# In[ ]:


# 统计无购买记录的用户
df_bzero = df_user[df_user['buy_num']==0]
# 输出购买数为0的总记录数
print (len(df_bzero))


# In[ ]:


# 删除无购买记录的用户
df_user = df_user[df_user['buy_num']!=0]


# In[ ]:


# 浏览购买转换比和点击购买转换比小于0.0005的用户为惰性用户
# 删除爬虫及惰性用户
bindex = df_user[df_user['buy_browse_ratio']<0.0005].index
print (len(bindex))
df_user.drop(bindex,axis=0,inplace=True)


# In[ ]:


# 点击购买转换比和点击购买转换比小于0.0005的用户为惰性用户
# 删除爬虫及惰性用户
cindex = df_user[df_user['buy_click_ratio']<0.0005].index
print (len(cindex))
df_user.drop(cindex,axis=0,inplace=True)


# In[ ]:


df_user.describe()


# In[ ]:


# 导入相关包
get_ipython().run_line_magic('matplotlib', 'inline')
# 绘图包
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


#定义文件名
ACTION_201602_FILE = "data/Data_Action_201602.csv"
ACTION_201603_FILE = "data/Data_Action_201603.csv"
ACTION_201604_FILE = "data/Data_Action_201604.csv"
COMMENT_FILE = "data/Data_Comment.csv"
PRODUCT_FILE = "data/Data_Product.csv"
USER_FILE = "data/Data_User.csv"
USER_TABLE_FILE = "data/User_table.csv"
ITEM_TABLE_FILE = "data/Item_table.csv"


# In[ ]:


# 提取购买(type=4)的下单行为数据
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[[
                "user_id", "sku_id", "type", "time"
            ]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4,为购买/下单
    df_ac = df_ac[df_ac['type'] == 4]

    return df_ac[["user_id", "sku_id", "time"]]


# In[ ]:


df_ac = []
df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))
df_ac = pd.concat(df_ac, ignore_index=True)


# In[ ]:


print(df_ac.dtypes) # 将time字段转换为datetime类型


# In[ ]:


# 将time字段转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])

# 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)


# In[ ]:


df_ac.head()


# In[ ]:


# 周一到周日每天购买用户个数
# 统计用户个数，nunique用于统计真值个数
df_user = df_ac.groupby('time')['user_id'].nunique()
# DataFrame可以通过set_index方法，可以设置索引
df_user = df_user.to_frame().reset_index()
df_user.columns = ['weekday', 'user_num']


# In[ ]:


# 周一到周日每天购买商品个数
df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['weekday', 'item_num']


# In[ ]:


# 周一到周日每天购买记录个数
df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['weekday', 'user_item_num']


# In[ ]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4

plt.bar(df_user['weekday'],
        df_user['user_num'],
        bar_width,
        alpha=opacity,
        color='c',
        label='user')
plt.bar(df_item['weekday'] + bar_width,
        df_item['item_num'],
        bar_width,
        alpha=opacity,
        color='g',
        label='item')
plt.bar(df_ui['weekday'] + bar_width * 2,
        df_ui['user_item_num'],
        bar_width,
        alpha=opacity,
        color='m',
        label='user_item')

plt.xlabel('weekday')
plt.ylabel('number')
plt.title('A Week Purchase Table')
plt.xticks(df_user['weekday'] + bar_width * 3 / 2., (1, 2, 3, 4, 5, 6, 7))
plt.tight_layout()
plt.legend(prop={'size': 10})

# 分析：周六，周日购买量较少，配送问题


# In[ ]:


df_ac = get_from_action_data(fname=ACTION_201602_FILE)

# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)


# In[ ]:


df_ac.head()


# In[ ]:


df_ac.tail()


# In[ ]:


df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['day', 'user_num']

df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['day', 'item_num']

df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['day', 'user_item_num']


# In[ ]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(df_user['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,10))

plt.bar(df_user['day'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['day']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')

plt.xlabel('day')
plt.ylabel('number')
plt.title('February Purchase Table')
plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# - 分析： 2月份5,6,7,8,9,10 这几天购买量非常少，原因可能是中国农历春节，快递不营业

# In[ ]:


df_ac = get_from_action_data(fname=ACTION_201603_FILE)

# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)


# In[ ]:


df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['day', 'user_num']

df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['day', 'item_num']

df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['day', 'user_item_num']


# In[ ]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(df_user['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,10))

plt.bar(df_user['day'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['day']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')

plt.xlabel('day')
plt.ylabel('number')
plt.title('March Purchase Table')
plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# In[ ]:


df_ac = get_from_action_data(fname=ACTION_201604_FILE)

# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac['time'] = pd.to_datetime(df_ac['time']).apply(lambda x: x.day)


# In[ ]:


df_user = df_ac.groupby('time')['user_id'].nunique()
df_user = df_user.to_frame().reset_index()
df_user.columns = ['day', 'user_num']

df_item = df_ac.groupby('time')['sku_id'].nunique()
df_item = df_item.to_frame().reset_index()
df_item.columns = ['day', 'item_num']

df_ui = df_ac.groupby('time', as_index=False).size()
df_ui = df_ui.to_frame().reset_index()
df_ui.columns = ['day', 'user_item_num']


# In[ ]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(df_user['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,10))

plt.bar(df_user['day'], df_user['user_num'], bar_width, 
        alpha=opacity, color='c', label='user')
plt.bar(df_item['day']+bar_width, df_item['item_num'], 
        bar_width, alpha=opacity, color='g', label='item')
plt.bar(df_ui['day']+bar_width*2, df_ui['user_item_num'], 
        bar_width, alpha=opacity, color='m', label='user_item')

plt.xlabel('day')
plt.ylabel('number')
plt.title('April Purchase Table')
plt.xticks(df_user['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# - 周一到周日各商品类别销售情况

# In[ ]:


# 从行为记录中提取商品类别数据
def get_from_action_data(fname, chunk_size=50000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[[
                "cate", "brand", "type", "time"
            ]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    # type=4,为购买
    df_ac = df_ac[df_ac['type'] == 4]

    return df_ac[["cate", "brand", "type", "time"]]


# In[ ]:


df_ac = []
df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))
df_ac = pd.concat(df_ac, ignore_index=True)


# In[ ]:


# 将time字段转换为datetime类型
df_ac['time'] = pd.to_datetime(df_ac['time'])

# 使用lambda匿名函数将时间time转换为星期(周一为1, 周日为７)
df_ac['time'] = df_ac['time'].apply(lambda x: x.weekday() + 1)


# In[ ]:


df_ac.head()


# In[ ]:


# 观察有几个类别商品
df_ac.groupby(df_ac['cate']).count()


# In[ ]:


# 周一到周日每天购买商品类别数量统计
df_product = df_ac['brand'].groupby([df_ac['time'], df_ac['cate']]).count()
# 把最内层的行索引还原成了列索引
df_product = df_product.unstack()
df_product.plot(kind='bar',
                title='Cate Purchase Table in a Week',
                figsize=(14, 10))


# - 分析：星期二买类别8的最多，星期天最少。

# In[ ]:


df_ac2 = get_from_action_data(fname=ACTION_201602_FILE)

# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac2['time'] = pd.to_datetime(df_ac2['time']).apply(lambda x: x.day)
df_ac3 = get_from_action_data(fname=ACTION_201603_FILE)

# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac3['time'] = pd.to_datetime(df_ac3['time']).apply(lambda x: x.day)
df_ac4 = get_from_action_data(fname=ACTION_201604_FILE)

# 将time字段转换为datetime类型并使用lambda匿名函数将时间time转换为天
df_ac4['time'] = pd.to_datetime(df_ac4['time']).apply(lambda x: x.day)


# In[ ]:


# cate 品类ID。brand 品牌ID
dc_cate2 = df_ac2[df_ac2['cate'] == 8]
dc_cate2 = dc_cate2['brand'].groupby(dc_cate2['time']).count()
dc_cate2 = dc_cate2.to_frame().reset_index()
dc_cate2.columns = ['day', 'product_num']

dc_cate3 = df_ac3[df_ac3['cate'] == 8]
dc_cate3 = dc_cate3['brand'].groupby(dc_cate3['time']).count()
dc_cate3 = dc_cate3.to_frame().reset_index()
dc_cate3.columns = ['day', 'product_num']

dc_cate4 = df_ac4[df_ac4['cate'] == 8]
dc_cate4 = dc_cate4['brand'].groupby(dc_cate4['time']).count()
dc_cate4 = dc_cate4.to_frame().reset_index()
dc_cate4.columns = ['day', 'product_num']


# In[ ]:


# 条形宽度
bar_width = 0.2
# 透明度
opacity = 0.4
# 天数
day_range = range(1,len(dc_cate3['day']) + 1, 1)
# 设置图片大小
plt.figure(figsize=(14,10))

plt.bar(dc_cate2['day'], dc_cate2['product_num'], bar_width, 
        alpha=opacity, color='c', label='February')
plt.bar(dc_cate3['day']+bar_width, dc_cate3['product_num'], 
        bar_width, alpha=opacity, color='g', label='March')
plt.bar(dc_cate4['day']+bar_width*2, dc_cate4['product_num'], 
        bar_width, alpha=opacity, color='m', label='April')

plt.xlabel('day')
plt.ylabel('number')
plt.title('Cate-8 Purchase Table')
plt.xticks(dc_cate3['day'] + bar_width * 3 / 2., day_range)
# plt.ylim(0, 80)
plt.tight_layout() 
plt.legend(prop={'size':9})


# In[ ]:


def spec_ui_action_data(fname, user_id, item_id, chunk_size=100000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[
                ["user_id", "sku_id", "type", "time"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    df_ac = df_ac[(df_ac['user_id'] == user_id) & (df_ac['sku_id'] == item_id)]

    return df_ac


# In[ ]:


def explore_user_item_via_time():
    user_id = 266079
    item_id = 138778
    df_ac = []
    df_ac.append(spec_ui_action_data(ACTION_201602_FILE, user_id, item_id))
    df_ac.append(spec_ui_action_data(ACTION_201603_FILE, user_id, item_id))
    df_ac.append(spec_ui_action_data(ACTION_201604_FILE, user_id, item_id))
    df_ac = pd.concat(df_ac, ignore_index=False)
    print(df_ac.sort_values(by='time'))


# In[ ]:


explore_user_item_via_time()


# In[ ]:


import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np


# In[ ]:


test = pd.read_csv('data/Data_Action_201602.csv')
test[['user_id','sku_id','model_id','type','cate','brand']] = test[['user_id','sku_id','model_id','type','cate','brand']].astype('float32')
test.dtypes
test.info() # 目的是float32位代替float64，节约内存


# In[ ]:


test = pd.read_csv('data/Data_Action_201602.csv')
test.dtypes
test.info()


# In[ ]:


action_1_path = 'data/Data_Action_201602.csv'
action_2_path = 'data/Data_Action_201603.csv'
action_3_path = 'data/Data_Action_201604.csv'

comment_path = 'data/Data_Comment.csv'
product_path = 'data/Data_Product.csv'
user_path = 'data/Data_User.csv'

# 评论日期
comment_date = [
    "2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29",
    "2016-03-07", "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04",
    "2016-04-11", "2016-04-15"
]


# In[ ]:


# 判断读入数据，哪种节约内存，速度快
def get_actions_0():
    action = pd.read_csv(action_1_path)
    return action


def get_actions_1():
    action = pd.read_csv(action_1_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate',
            'brand']] = action[[
                'user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand'
            ]].astype('float32')
    return action


def get_actions_2():
    action = pd.read_csv(action_2_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate',
            'brand']] = action[[
                'user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand'
            ]].astype('float32')

    return action


def get_actions_3():
    action = pd.read_csv(action_3_path)
    action[['user_id', 'sku_id', 'model_id', 'type', 'cate',
            'brand']] = action[[
                'user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand'
            ]].astype('float32')

    return action


def get_actions_10():

    reader = pd.read_csv(action_1_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate',
            'brand']] = reader[[
                'user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand'
            ]].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


def get_actions_20():

    reader = pd.read_csv(action_2_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate',
            'brand']] = reader[[
                'user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand'
            ]].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


def get_actions_30():

    reader = pd.read_csv(action_3_path, iterator=True)
    reader[['user_id', 'sku_id', 'model_id', 'type', 'cate',
            'brand']] = reader[[
                'user_id', 'sku_id', 'model_id', 'type', 'cate', 'brand'
            ]].astype('float32')
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(50000)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")
    action = pd.concat(chunks, ignore_index=True)

    return action


# 读取并拼接所有行为记录文件
def get_all_action():
    action_1 = get_actions_1()
    action_2 = get_actions_2()
    action_3 = get_actions_3()
    actions = pd.concat([action_1, action_2, action_3])  # type: pd.DataFrame

    return actions


# 获取某个时间段的行为记录，大于等于起始时间，小于终止时间
def get_actions(start_date, end_date, all_actions):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    actions = all_actions[(all_actions.time >= start_date)
                          & (all_actions.time < end_date)].copy()
    return actions


# In[ ]:


from sklearn import preprocessing


# 获取基本的用户信息
def get_basic_user_feat():
    # 针对年龄的中文字符问题处理，首先是读入的时候编码，
    # 填充空值，然后将其数值化，最后独热编码，此外对于sex也进行了数值类型转换
    user = pd.read_csv(user_path, encoding='gbk')
    # 剔除空值
    user.dropna(axis=0, how='any', inplace=True)
    user['sex'] = user['sex'].astype(int)
    user['age'] = user['age'].astype(int)
    le = preprocessing.LabelEncoder()
    age_df = le.fit_transform(user['age'])
    #     print list(le.classes_)

    age_df = pd.get_dummies(age_df, prefix='age')
    sex_df = pd.get_dummies(user['sex'], prefix='sex')
    # 用户等级
    user_lv_df = pd.get_dummies(user['user_lv_cd'], prefix='user_lv_cd')
    user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
    return user


# In[ ]:


user = pd.read_csv(user_path, encoding='gbk')
# 判断是否文件中有空值，False代表没有空值，True代表有空值
user.isnull().any()


# In[ ]:


user[user.isnull().values==True] # 检查所有空值的列


# In[ ]:


user.dropna(axis=0, how='any',inplace=True) # 按行删除
user.isnull().any() # False代表没有缺失值


# In[ ]:


def get_basic_product_feat():
    product = pd.read_csv(product_path)
    attr1_df = pd.get_dummies(product["a1"], prefix="a1")
    attr2_df = pd.get_dummies(product["a2"], prefix="a2")
    attr3_df = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat(
        [product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df],
        axis=1)
    return product


# In[ ]:


def get_comments_product_feat(end_date):
    comments = pd.read_csv(comment_path)
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    comments = comments[comments.dt == comment_date_begin]
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
    # 为了防止某个时间段不具备评论数为0的情况（测试集出现过这种情况）
    for i in range(0, 5):
        if 'comment_num_' + str(i) not in df.columns:
            df['comment_num_' + str(i)] = 0
    df = df[[
        'comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3',
        'comment_num_4'
    ]]

    comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
    #del comments['dt']
    #del comments['comment_num']
    comments = comments[[
        'sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0',
        'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4'
    ]]
    return comments


# In[ ]:


train_start_date = '2016-02-01'
train_end_date = datetime.strptime(train_start_date,
                                   '%Y-%m-%d') + timedelta(days=3)
train_end_date = train_end_date.strftime('%Y-%m-%d')

day = 3

start_date = datetime.strptime(train_end_date,
                               '%Y-%m-%d') - timedelta(days=day)
start_date = start_date.strftime('%Y-%m-%d')


# In[ ]:


'''
comment_date = [
    "2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29",
    "2016-03-07", "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04",
    "2016-04-11", "2016-04-15"
]
'''
comments = pd.read_csv(comment_path)
comment_date_end = train_end_date
comment_date_begin = comment_date[0]
# 保证开始时间比结束时间小
for date in reversed(comment_date):
    if date < comment_date_end:
        comment_date_begin = date
        break
# 可以更改时间段
comments = comments[comments.dt == comment_date_begin]
df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
# 为了防止某个时间段不具备评论数为0的情况（测试集出现过这种情况）
for i in range(0, 5):
    if 'comment_num_' + str(i) not in df.columns:
        df['comment_num_' + str(i)] = 0
df = df[[
    'comment_num_0', 'comment_num_1', 'comment_num_2', 'comment_num_3',
    'comment_num_4'
]]

comments = pd.concat([comments, df], axis=1)

comments = comments[[
    'sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0',
    'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4'
]]
comments.head()


# In[ ]:


def get_action_feat(start_date, end_date, all_actions, i):
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'sku_id', 'cate', 'type']]
    # 不同时间累积的行为计数（3,5,7,10,15,21,30）
    # prefix:设置列名前缀
    df = pd.get_dummies(actions['type'], prefix='action_before_%s' % i)
    before_date = 'action_before_%s' % i
    actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
    # 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
    actions = actions.groupby(['user_id', 'sku_id', 'cate'],
                              as_index=False).sum()
    # 分组统计，用户-类别，不同用户对不同商品类别的行为计数
    user_cate = actions.groupby(['user_id', 'cate'], as_index=False).sum()
    # 删除重复数据
    del user_cate['sku_id']
    del user_cate['type']
    actions = pd.merge(actions, user_cate, how='left', on=['user_id', 'cate'])
    #本类别下其他商品点击量
    # 前述两种分组含有相同名称的不同行为的计数，系统会自动针对名称调整添加后缀,x,y，
    # 所以这里作差统计的是同一类别下其他商品的行为计数
    # 类别计数-某一商品计数
    actions[before_date +
            '_1.0_y'] = actions[before_date + '_1.0_y'] - actions[before_date +
                                                                  '_1.0_x']
    actions[before_date +
            '_2.0_y'] = actions[before_date + '_2.0_y'] - actions[before_date +
                                                                  '_2.0_x']
    actions[before_date +
            '_3.0_y'] = actions[before_date + '_3.0_y'] - actions[before_date +
                                                                  '_3.0_x']
    actions[before_date +
            '_4.0_y'] = actions[before_date + '_4.0_y'] - actions[before_date +
                                                                  '_4.0_x']
    actions[before_date +
            '_5.0_y'] = actions[before_date + '_5.0_y'] - actions[before_date +
                                                                  '_5.0_x']
    actions[before_date +
            '_6.0_y'] = actions[before_date + '_6.0_y'] - actions[before_date +
                                                                  '_6.0_x']
    # 统计用户对不同类别下商品计数与该类别下商品行为计数均值（对时间）的差值
    actions[before_date + 'minus_mean_1'] = actions[before_date + '_1.0_x'] - (
        actions[before_date + '_1.0_x'] / i)
    actions[before_date + 'minus_mean_2'] = actions[before_date + '_2.0_x'] - (
        actions[before_date + '_2.0_x'] / i)
    actions[before_date + 'minus_mean_3'] = actions[before_date + '_3.0_x'] - (
        actions[before_date + '_3.0_x'] / i)
    actions[before_date + 'minus_mean_4'] = actions[before_date + '_4.0_x'] - (
        actions[before_date + '_4.0_x'] / i)
    actions[before_date + 'minus_mean_5'] = actions[before_date + '_5.0_x'] - (
        actions[before_date + '_5.0_x'] / i)
    actions[before_date + 'minus_mean_6'] = actions[before_date + '_6.0_x'] - (
        actions[before_date + '_6.0_x'] / i)
    del actions['type']
    # 保留cate特征
    #     del actions['cate']

    return actions


# In[ ]:


all_actions = get_all_action()

actions = get_actions(start_date, train_end_date, all_actions)
actions = actions[['user_id', 'sku_id', 'cate', 'type']]
# 不同时间累积的行为计数（3,5,7,10,15,21,30）
df = pd.get_dummies(actions['type'], prefix='action_before_%s' % 3)
before_date = 'action_before_%s' % 3
actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
# 分组统计，用户-类别-商品,不同用户对不同类别下商品的行为计数
actions = actions.groupby(['user_id', 'sku_id', 'cate'], as_index=False).sum()
actions.head(20)
# 在3天之内，行为/type为2，action_before_3_1.0


# In[ ]:


# 某一个客户对第4大类别3天内做了为”1“的行为的次数为48
user_cate = actions.groupby(['user_id', 'cate'], as_index=False).sum()
del user_cate['sku_id']
del user_cate['type']
user_cate.head()


# In[ ]:


actions = pd.merge(actions, user_cate, how='left', on=['user_id', 'cate'])
actions.head()
# x指的是特定的商品，”sku_id“；y指的是品类”cate“


# In[ ]:


# 差异/区别/比重，在某个品类里面，除了某个特定的商品，其他的商品是什么情况
actions[before_date +
        '_1_y'] = actions[before_date + '_1.0_y'] - actions[before_date +
                                                            '_1.0_x']
actions.head()


# In[ ]:


all_actions


# In[ ]:


# 获取累积用户行为特征
def get_accumulate_user_feat(end_date, all_actions, day):
    start_date = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=day)
    start_date = start_date.strftime('%Y-%m-%d')
    before_date = 'user_action_%s' % day
    # 整体特征，可忽略
    feature = [
        'user_id', before_date + '_1', before_date + '_2', before_date + '_3',
        before_date + '_4', before_date + '_5', before_date + '_6',
        before_date + '_1_ratio', before_date + '_2_ratio',
        before_date + '_3_ratio', before_date + '_5_ratio',
        before_date + '_6_ratio', before_date + '_1_mean',
        before_date + '_2_mean', before_date + '_3_mean',
        before_date + '_4_mean', before_date + '_5_mean',
        before_date + '_6_mean', before_date + '_1_std',
        before_date + '_2_std', before_date + '_3_std', before_date + '_4_std',
        before_date + '_5_std', before_date + '_6_std'
    ]
    # 获取开始到结束时间内所有的行为数据
    actions = get_actions(start_date, end_date, all_actions)
    df = pd.get_dummies(actions['type'], prefix=before_date)
    # 添加日期列
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
  
    actions = pd.concat([actions[['user_id', 'date']], df], axis=1)
    
    actions[before_date +
            '_1_ratio'] = np.log(1 + actions[before_date + '_4.0']) - np.log(
                1 + actions[before_date + '_1.0'])
    actions[before_date +
            '_2_ratio'] = np.log(1 + actions[before_date + '_4.0']) - np.log(
                1 + actions[before_date + '_2.0'])
    actions[before_date +
            '_3_ratio'] = np.log(1 + actions[before_date + '_4.0']) - np.log(
                1 + actions[before_date + '_3.0'])
    actions[before_date +
            '_5_ratio'] = np.log(1 + actions[before_date + '_4.0']) - np.log(
                1 + actions[before_date + '_5.0'])
    actions[before_date +
            '_6_ratio'] = np.log(1 + actions[before_date + '_4.0']) - np.log(
                1 + actions[before_date + '_6.0'])
    # 均值
    actions[before_date + '_1_mean'] = actions[before_date + '_1.0'] / day
    actions[before_date + '_2_mean'] = actions[before_date + '_2.0'] / day
    actions[before_date + '_3_mean'] = actions[before_date + '_3.0'] / day
    actions[before_date + '_4_mean'] = actions[before_date + '_4.0'] / day
    actions[before_date + '_5_mean'] = actions[before_date + '_5.0'] / day
    actions[before_date + '_6_mean'] = actions[before_date + '_6.0'] / day
    # actions = pd.merge(actions, actions_date, how='left', on='user_id')
    # actions = actions[feature]
    return actions


# In[ ]:


train_start_date = '2016-02-01'
train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
train_end_date = train_end_date.strftime('%Y-%m-%d')
day = 3

start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
start_date = start_date.strftime('%Y-%m-%d')
before_date = 'user_action_%s' % day


# In[ ]:


before_date


# In[ ]:


# 取3天的时间判断
print (start_date)
print (train_end_date)


# In[ ]:


all_actions.shape


# In[ ]:


actions = get_actions(start_date, train_end_date, all_actions)
actions.shape


# In[ ]:


actions.head()


# In[ ]:


df = pd.get_dummies(actions['type'], prefix=before_date)
df.head()


# In[ ]:


actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
actions = pd.concat([actions[['user_id', 'date']], df], axis=1)
actions_date = actions.groupby(['user_id', 'date']).sum()
actions_date.head()


# In[ ]:


actions_date = actions_date.unstack()
actions_date.fillna(0, inplace=True)
actions_date.head(3)


# In[ ]:


actions = actions.groupby(['user_id'], as_index=False).sum()
actions.head()
# 对于用户来说，什么样的行为才能购买


# In[ ]:


# _4.0：代表的是购买，转化率：logA-logB = logA/B ，user_action_3_1_ratio
actions[before_date + '_1_ratio'] =  np.log(1 + actions[before_date + '_4.0']) - np.log(1 + actions[before_date +'_1.0'])
actions.head()


# In[ ]:


# 均值
actions[before_date + '_1_mean'] = actions[before_date + '_1.0'] / day
actions.head()


# In[ ]:


def get_recent_user_feat(end_date, all_actions):

    actions_3 = get_accumulate_user_feat(end_date, all_actions,
                                         3)  # 通过终止时间往前推3天
    actions_30 = get_accumulate_user_feat(end_date, all_actions,
                                          30)  # 通过终止时间往前推30天
    actions = pd.merge(actions_3, actions_30, how='left', on='user_id')
    del actions_3
    del actions_30
    # 一个月内用户除去最近三天的行为占据一个月的行为的比重
    actions['recent_action1'] = np.log(1 + actions['user_action_30_1.0'] -
                                       actions['user_action_3_1.0']) - np.log(
                                           1 + actions['user_action_30_1.0'])
    actions['recent_action2'] = np.log(1 + actions['user_action_30_2.0'] -
                                       actions['user_action_3_2.0']) - np.log(
                                           1 + actions['user_action_30_2.0'])
    actions['recent_action3'] = np.log(1 + actions['user_action_30_3.0'] -
                                       actions['user_action_3_3.0']) - np.log(
                                           1 + actions['user_action_30_3.0'])
    actions['recent_action4'] = np.log(1 + actions['user_action_30_4.0'] -
                                       actions['user_action_3_4.0']) - np.log(
                                           1 + actions['user_action_30_4.0'])
    actions['recent_action5'] = np.log(1 + actions['user_action_30_5.0'] -
                                       actions['user_action_3_5.0']) - np.log(
                                           1 + actions['user_action_30_5.0'])
    actions['recent_action6'] = np.log(1 + actions['user_action_30_6.0'] -
                                       actions['user_action_3_6.0']) - np.log(
                                           1 + actions['user_action_30_6.0'])

    return actions


# In[ ]:


# 增加了用户对不同类别的交互特征
def get_user_cate_feature(start_date, end_date, all_actions):
    actions = get_actions(start_date, end_date, all_actions)
    actions = actions[['user_id', 'cate', 'type']]
    df = pd.get_dummies(actions['type'], prefix='type')
    actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
    actions = actions.groupby(['user_id', 'cate']).sum()
    actions = actions.unstack()
    # 双索引互换
    actions.columns = actions.columns.swaplevel(0, 1)
    # 删除索引（默认0）
    actions.columns = actions.columns.droplevel()
    # 类别行为特征举例
    actions.columns = [
        'cate_4_type1', 'cate_5_type1', 'cate_6_type1', 'cate_7_type1',
        'cate_8_type1', 'cate_9_type1', 'cate_10_type1', 'cate_11_type1',
        'cate_4_type2', 'cate_5_type2', 'cate_6_type2', 'cate_7_type2',
        'cate_8_type2', 'cate_9_type2', 'cate_10_type2', 'cate_11_type2',
        'cate_4_type3', 'cate_5_type3', 'cate_6_type3', 'cate_7_type3',
        'cate_8_type3', 'cate_9_type3', 'cate_10_type3', 'cate_11_type3',
        'cate_4_type4', 'cate_5_type4', 'cate_6_type4', 'cate_7_type4',
        'cate_8_type4', 'cate_9_type4', 'cate_10_type4', 'cate_11_type4',
        'cate_4_type5', 'cate_5_type5', 'cate_6_type5', 'cate_7_type5',
        'cate_8_type5', 'cate_9_type5', 'cate_10_type5', 'cate_11_type5',
        'cate_4_type6', 'cate_5_type6', 'cate_6_type6', 'cate_7_type6',
        'cate_8_type6', 'cate_9_type6', 'cate_10_type6', 'cate_11_type6'
    ]
    # 空值填充0
    actions = actions.fillna(0)
    # 类别行为统计
    actions['cate_action_sum'] = actions.sum(axis=1)
    # 该类别商品所占比例
    actions['cate8_percentage'] = (
        actions['cate_8_type1'] + actions['cate_8_type2'] +
        actions['cate_8_type3'] + actions['cate_8_type4'] +
        actions['cate_8_type5'] +
        actions['cate_8_type6']) / actions['cate_action_sum']
    actions['cate4_percentage'] = (
        actions['cate_4_type1'] + actions['cate_4_type2'] +
        actions['cate_4_type3'] + actions['cate_4_type4'] +
        actions['cate_4_type5'] +
        actions['cate_4_type6']) / actions['cate_action_sum']
    actions['cate5_percentage'] = (
        actions['cate_5_type1'] + actions['cate_5_type2'] +
        actions['cate_5_type3'] + actions['cate_5_type4'] +
        actions['cate_5_type5'] +
        actions['cate_5_type6']) / actions['cate_action_sum']
    actions['cate6_percentage'] = (
        actions['cate_6_type1'] + actions['cate_6_type2'] +
        actions['cate_6_type3'] + actions['cate_6_type4'] +
        actions['cate_6_type5'] +
        actions['cate_6_type6']) / actions['cate_action_sum']
    actions['cate7_percentage'] = (
        actions['cate_7_type1'] + actions['cate_7_type2'] +
        actions['cate_7_type3'] + actions['cate_7_type4'] +
        actions['cate_7_type5'] +
        actions['cate_7_type6']) / actions['cate_action_sum']
    actions['cate9_percentage'] = (
        actions['cate_9_type1'] + actions['cate_9_type2'] +
        actions['cate_9_type3'] + actions['cate_9_type4'] +
        actions['cate_9_type5'] +
        actions['cate_9_type6']) / actions['cate_action_sum']
    actions['cate10_percentage'] = (
        actions['cate_10_type1'] + actions['cate_10_type2'] +
        actions['cate_10_type3'] + actions['cate_10_type4'] +
        actions['cate_10_type5'] +
        actions['cate_10_type6']) / actions['cate_action_sum']
    actions['cate11_percentage'] = (
        actions['cate_11_type1'] + actions['cate_11_type2'] +
        actions['cate_11_type3'] + actions['cate_11_type4'] +
        actions['cate_11_type5'] +
        actions['cate_11_type6']) / actions['cate_action_sum']
    # 某类别下某行为，占其他类型行为的比例
    actions['cate8_type1_percentage'] = np.log(
        1 + actions['cate_8_type1']) - np.log(
            1 + actions['cate_8_type1'] + actions['cate_4_type1'] +
            actions['cate_5_type1'] + actions['cate_6_type1'] +
            actions['cate_7_type1'] + actions['cate_9_type1'] +
            actions['cate_10_type1'] + actions['cate_11_type1'])

    actions['cate8_type2_percentage'] = np.log(
        1 + actions['cate_8_type2']) - np.log(
            1 + actions['cate_8_type2'] + actions['cate_4_type2'] +
            actions['cate_5_type2'] + actions['cate_6_type2'] +
            actions['cate_7_type2'] + actions['cate_9_type2'] +
            actions['cate_10_type2'] + actions['cate_11_type2'])
    actions['cate8_type3_percentage'] = np.log(
        1 + actions['cate_8_type3']) - np.log(
            1 + actions['cate_8_type3'] + actions['cate_4_type3'] +
            actions['cate_5_type3'] + actions['cate_6_type3'] +
            actions['cate_7_type3'] + actions['cate_9_type3'] +
            actions['cate_10_type3'] + actions['cate_11_type3'])
    actions['cate8_type4_percentage'] = np.log(
        1 + actions['cate_8_type4']) - np.log(
            1 + actions['cate_8_type4'] + actions['cate_4_type4'] +
            actions['cate_5_type4'] + actions['cate_6_type4'] +
            actions['cate_7_type4'] + actions['cate_9_type4'] +
            actions['cate_10_type4'] + actions['cate_11_type4'])
    actions['cate8_type5_percentage'] = np.log(
        1 + actions['cate_8_type5']) - np.log(
            1 + actions['cate_8_type5'] + actions['cate_4_type5'] +
            actions['cate_5_type5'] + actions['cate_6_type5'] +
            actions['cate_7_type5'] + actions['cate_9_type5'] +
            actions['cate_10_type5'] + actions['cate_11_type5'])
    actions['cate8_type6_percentage'] = np.log(
        1 + actions['cate_8_type6']) - np.log(
            1 + actions['cate_8_type6'] + actions['cate_4_type6'] +
            actions['cate_5_type6'] + actions['cate_6_type6'] +
            actions['cate_7_type6'] + actions['cate_9_type6'] +
            actions['cate_10_type6'] + actions['cate_11_type6'])
    actions['user_id'] = actions.index
    actions = actions[[
        'user_id', 'cate8_percentage', 'cate4_percentage', 'cate5_percentage',
        'cate6_percentage', 'cate7_percentage', 'cate9_percentage',
        'cate10_percentage', 'cate11_percentage', 'cate8_type1_percentage',
        'cate8_type2_percentage', 'cate8_type3_percentage',
        'cate8_type4_percentage', 'cate8_type5_percentage',
        'cate8_type6_percentage'
    ]]
    return actions


# In[ ]:


train_start_date = '2016-02-01'
train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
train_end_date = train_end_date.strftime('%Y-%m-%d')
day = 3

start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
start_date = start_date.strftime('%Y-%m-%d')

print (start_date)
print (train_end_date)


# In[ ]:


actions = get_actions(start_date, train_end_date, all_actions)
actions = actions[['user_id', 'cate', 'type']]
actions.head()


# In[ ]:


df = pd.get_dummies(actions['type'], prefix='type')
actions = pd.concat([actions[['user_id', 'cate']], df], axis=1)
actions = actions.groupby(['user_id', 'cate']).sum()
actions.head()


# In[ ]:


actions = actions.unstack() # 花括号结构变成表结构
actions.head()


# In[ ]:


actions.columns


# In[ ]:


actions.columns = actions.columns.swaplevel(0, 1)#接受两个级别编号或名称,并返回一个互换了级别的新对象(但数据不会发生变化)
actions.columns


# In[ ]:


actions.columns = actions.columns.droplevel()
actions.columns


# In[ ]:


actions.columns = [
    'cate_4_type1', 'cate_5_type1', 'cate_6_type1', 'cate_7_type1',
    'cate_8_type1', 'cate_9_type1', 'cate_10_type1', 'cate_11_type1',
    'cate_4_type2', 'cate_5_type2', 'cate_6_type2', 'cate_7_type2',
    'cate_8_type2', 'cate_9_type2', 'cate_10_type2', 'cate_11_type2',
    'cate_4_type3', 'cate_5_type3', 'cate_6_type3', 'cate_7_type3',
    'cate_8_type3', 'cate_9_type3', 'cate_10_type3', 'cate_11_type3',
    'cate_4_type4', 'cate_5_type4', 'cate_6_type4', 'cate_7_type4',
    'cate_8_type4', 'cate_9_type4', 'cate_10_type4', 'cate_11_type4',
    'cate_4_type5', 'cate_5_type5', 'cate_6_type5', 'cate_7_type5',
    'cate_8_type5', 'cate_9_type5', 'cate_10_type5', 'cate_11_type5',
    'cate_4_type6', 'cate_5_type6', 'cate_6_type6', 'cate_7_type6',
    'cate_8_type6', 'cate_9_type6', 'cate_10_type6', 'cate_11_type6'
]
actions.columns


# In[ ]:


actions = actions.fillna(0) # 拿0填充，因为用户没有行为
actions['cate_action_sum'] = actions.sum(axis=1)
actions.head()
# 一个用户对第4大类，执行了行为1的动作


# In[ ]:


# 第第八大类的6种行为，各自的占比
actions['cate8_percentage'] = (
    actions['cate_8_type1'] + actions['cate_8_type2'] + actions['cate_8_type3']
    + actions['cate_8_type4'] + actions['cate_8_type5'] +
    actions['cate_8_type6']) / actions['cate_action_sum']
actions.head()


# In[ ]:


actions['cate8_type1_percentage'] = np.log(
        1 + actions['cate_8_type1']) - np.log(
            1 + actions['cate_8_type1'] + actions['cate_4_type1'] +
            actions['cate_5_type1'] + actions['cate_6_type1'] +
            actions['cate_7_type1'] + actions['cate_9_type1'] +
            actions['cate_10_type1'] + actions['cate_11_type1'])
actions.head()


# In[ ]:


# 创建累计商品特征
def get_accumulate_product_feat(start_date, end_date, all_actions):
    # 我们要创建的特征
    feature = [
        'sku_id', 'product_action_1', 'product_action_2', 'product_action_3',
        'product_action_4', 'product_action_5', 'product_action_6',
        'product_action_1_ratio', 'product_action_2_ratio',
        'product_action_3_ratio', 'product_action_5_ratio',
        'product_action_6_ratio', 'product_action_1_mean',
        'product_action_2_mean', 'product_action_3_mean',
        'product_action_4_mean', 'product_action_5_mean',
        'product_action_6_mean', 'product_action_1_std',
        'product_action_2_std', 'product_action_3_std', 'product_action_4_std',
        'product_action_5_std', 'product_action_6_std'
    ]

    actions = get_actions(start_date, end_date, all_actions)
    df = pd.get_dummies(actions['type'], prefix='product_action')
    # 按照商品-日期分组，计算某个时间段该商品的各项行为的标准差
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    actions = pd.concat([actions[['sku_id', 'date']], df], axis=1)

    actions = actions.groupby(['sku_id'], as_index=False).sum()
    # 生成时间段
    days_interal = (datetime.strptime(end_date, '%Y-%m-%d') -
                    datetime.strptime(start_date, '%Y-%m-%d')).days
    # 某商品的购买和各种行为的比率
    actions['product_action_1_ratio'] = np.log(
        1 +
        actions['product_action_4.0']) - np.log(1 +
                                                actions['product_action_1.0'])
    actions['product_action_2_ratio'] = np.log(
        1 +
        actions['product_action_4.0']) - np.log(1 +
                                                actions['product_action_2.0'])
    actions['product_action_3_ratio'] = np.log(
        1 +
        actions['product_action_4.0']) - np.log(1 +
                                                actions['product_action_3.0'])
    actions['product_action_5_ratio'] = np.log(
        1 +
        actions['product_action_4.0']) - np.log(1 +
                                                actions['product_action_5.0'])
    actions['product_action_6_ratio'] = np.log(
        1 +
        actions['product_action_4.0']) - np.log(1 +
                                                actions['product_action_6.0'])
    # 计算各种行为的均值
    actions[
        'product_action_1_mean'] = actions['product_action_1.0'] / days_interal
    actions[
        'product_action_2_mean'] = actions['product_action_2.0'] / days_interal
    actions[
        'product_action_3_mean'] = actions['product_action_3.0'] / days_interal
    actions[
        'product_action_4_mean'] = actions['product_action_4.0'] / days_interal
    actions[
        'product_action_5_mean'] = actions['product_action_5.0'] / days_interal
    actions[
        'product_action_6_mean'] = actions['product_action_6.0'] / days_interal
    #actions = pd.merge(actions, actions_date, how='left', on='sku_id')
    #actions = actions[feature]
    return actions


# In[ ]:


train_start_date = '2016-02-01'
train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
train_end_date = train_end_date.strftime('%Y-%m-%d')
day = 3

start_date = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=day)
start_date = start_date.strftime('%Y-%m-%d')

print (start_date)
print (train_end_date)


# In[ ]:


actions = get_actions(start_date, train_end_date, all_actions)
df = pd.get_dummies(actions['type'], prefix='product_action')

actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
actions = pd.concat([actions[['sku_id', 'date']], df], axis=1)
actions.head()


# In[ ]:


actions = actions.groupby(['sku_id'], as_index=False).sum()
actions.head()


# In[ ]:


days_interal = (datetime.strptime(train_end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
days_interal


# In[ ]:


actions['product_action_1_ratio'] =  np.log(1 + actions['product_action_4.0']) - np.log(1 + actions['product_action_1.0'])
actions.head()


# In[ ]:


# 各个时间段下各类商品的购买转化率，标准差，均值
def get_accumulate_cate_feat(start_date, end_date, all_actions):
    # 所要创建的特征名称
    feature = [
        'cate', 'cate_action_1', 'cate_action_2', 'cate_action_3',
        'cate_action_4', 'cate_action_5', 'cate_action_6',
        'cate_action_1_ratio', 'cate_action_2_ratio', 'cate_action_3_ratio',
        'cate_action_5_ratio', 'cate_action_6_ratio', 'cate_action_1_mean',
        'cate_action_2_mean', 'cate_action_3_mean', 'cate_action_4_mean',
        'cate_action_5_mean', 'cate_action_6_mean', 'cate_action_1_std',
        'cate_action_2_std', 'cate_action_3_std', 'cate_action_4_std',
        'cate_action_5_std', 'cate_action_6_std'
    ]
    actions = get_actions(start_date, end_date, all_actions)
    actions['date'] = pd.to_datetime(actions['time']).apply(lambda x: x.date())
    df = pd.get_dummies(actions['type'], prefix='cate_action')
    actions = pd.concat([actions[['cate', 'date']], df], axis=1)
    # 按照类别分组，统计各个商品类别下行为的转化率
    actions = actions.groupby(['cate'], as_index=False).sum()
    days_interal = (datetime.strptime(end_date, '%Y-%m-%d') -
                    datetime.strptime(start_date, '%Y-%m-%d')).days

    actions['cate_action_1_ratio'] = (np.log(1 + actions['cate_action_4.0']) -
                                      np.log(1 + actions['cate_action_1.0']))
    actions['cate_action_2_ratio'] = (np.log(1 + actions['cate_action_4.0']) -
                                      np.log(1 + actions['cate_action_2.0']))
    actions['cate_action_3_ratio'] = (np.log(1 + actions['cate_action_4.0']) -
                                      np.log(1 + actions['cate_action_3.0']))
    actions['cate_action_5_ratio'] = (np.log(1 + actions['cate_action_4.0']) -
                                      np.log(1 + actions['cate_action_5.0']))
    actions['cate_action_6_ratio'] = (np.log(1 + actions['cate_action_4.0']) -
                                      np.log(1 + actions['cate_action_6.0']))
    # 按照类别分组，统计各个商品类别下行为在一段时间的均值
    actions['cate_action_1_mean'] = actions['cate_action_1.0'] / days_interal
    actions['cate_action_2_mean'] = actions['cate_action_2.0'] / days_interal
    actions['cate_action_3_mean'] = actions['cate_action_3.0'] / days_interal
    actions['cate_action_4_mean'] = actions['cate_action_4.0'] / days_interal
    actions['cate_action_5_mean'] = actions['cate_action_5.0'] / days_interal
    actions['cate_action_6_mean'] = actions['cate_action_6.0'] / days_interal
    #actions = pd.merge(actions, actions_date, how ='left',on='cate')
    #actions = actions[feature]
    return actions


# In[ ]:


# 获取标签
def get_labels(start_date, end_date, all_actions):
    actions = get_actions(start_date, end_date, all_actions)

    # 修改为预测购买了商品8的用户预测
    actions = actions[(actions['type'] == 4) & (actions['cate'] == 8)]

    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    actions['label'] = 1
    actions = actions[['user_id', 'sku_id', 'label']]
    return actions


# In[ ]:


train_start_date = '2016-03-01'
train_actions = None
all_actions = get_all_action()
print ("get all actions!")


# In[ ]:


all_actions.head()


# In[ ]:


all_actions.info()


# In[ ]:


all_actions.shape


# In[ ]:


# 用户基本特征
user = get_basic_user_feat()
print ('get_basic_user_feat finsihed')


# In[ ]:


user.head()


# In[ ]:


# 商品基本特征
product = get_basic_product_feat()
print ('get_basic_product_feat finsihed')


# In[ ]:


product.head()


# In[ ]:


# 设置起始时间
train_start_date = '2016-03-01'
train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
train_end_date


# In[ ]:


train_end_date = train_end_date.strftime('%Y-%m-%d')
# 修正prod_acc,cate_acc的时间跨度
start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
start_days = start_days.strftime('%Y-%m-%d')
print (train_end_date)


# In[ ]:


start_days


# In[ ]:


# 构造行为特征
def make_actions(user, product, all_actions, train_start_date):
    train_end_date = datetime.strptime(train_start_date,
                                       '%Y-%m-%d') + timedelta(days=3)
    train_end_date = train_end_date.strftime('%Y-%m-%d')
    # 修正prod_acc,cate_acc的时间跨度
    start_days = datetime.strptime(train_end_date,
                                   '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    print(train_end_date)
    # 用户近期行为特征
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print('get_recent_user_feat finsihed')
    # 用户对同类别下各种商品的行为特征
    user_cate = get_user_cate_feature(train_start_date, train_end_date,
                                      all_actions)
    print('get_user_cate_feature finished')
    # 累计商品特征
    product_acc = get_accumulate_product_feat(start_days, train_end_date,
                                              all_actions)
    print('get_accumulate_product_feat finsihed')
    # 类别特征
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date,
                                        all_actions)
    print('get_accumulate_cate_feat finsihed')
    # 评论特征
    comment_acc = get_comments_product_feat(train_end_date)
    print('get_comments_product_feat finished')
    # 标记
    test_start_date = train_end_date
    test_end_date = datetime.strptime(test_start_date,
                                      '%Y-%m-%d') + timedelta(days=5)
    test_end_date = test_end_date.strftime('%Y-%m-%d')
    # 标签
    labels = get_labels(test_start_date, test_end_date, all_actions)
    print("get labels")

    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date,
                                       '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            # 按照不同日期获取行为特征
            actions = get_action_feat(start_days, train_end_date, all_actions,
                                      i)
        else:
            # 注意这里的拼接key
            actions = pd.merge(actions,
                               get_action_feat(start_days, train_end_date,
                                               all_actions, i),
                               how='left',
                               on=['user_id', 'sku_id', 'cate'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    user_cate.index.name = ""
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    # 主要是填充拼接商品基本特征、评论特征、标签之后的空值
    actions = actions.fillna(0)
    #     return actions
    # 采样
    action_postive = actions[actions['label'] == 1]
    action_negative = actions[actions['label'] == 0]
    del actions
    neg_len = len(action_postive) * 10
    # 随机抽取样本
    action_negative = action_negative.sample(n=neg_len)
    action_sample = pd.concat([action_postive, action_negative],
                              ignore_index=True)

    return action_sample


# In[ ]:


#  制作训练集
def make_train_set(train_start_date, setNums ,f_path, all_actions):
    train_actions = None
    # 用户基本特征
    user = get_basic_user_feat()
    print ('get_basic_user_feat finsihed')
    # 商品基本特征
    product = get_basic_product_feat()
    print ('get_basic_product_feat finsihed')
    # 滑窗,构造多组训练集/验证集
    for i in range(setNums):
        print (train_start_date)
        if train_actions is None:
            train_actions = make_actions(user, product, all_actions, train_start_date)
        else:
            train_actions = pd.concat([train_actions, make_actions(user, product, all_actions, train_start_date)],
                                          ignore_index=True)
        # 接下来每次移动一天
        train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=1)
        train_start_date = train_start_date.strftime('%Y-%m-%d')
        print ("round {0}/{1} over!".format(i+1, setNums))

    train_actions.to_csv(f_path, index=False)


# In[ ]:


train_start_date = '2016-02-01'
train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=3)
train_end_date

train_end_date = train_end_date.strftime('%Y-%m-%d')
# 修正prod_acc,cate_acc的时间跨度
start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
start_days = start_days.strftime('%Y-%m-%d')
print (train_end_date)


# In[ ]:


user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
print ('get_user_cate_feature finished')


# In[ ]:


product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
print ('get_accumulate_product_feat finsihed')


# In[ ]:


cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
print ('get_accumulate_cate_feat finsihed')


# In[ ]:


# 训练集
train_start_date = '2016-02-01'
make_train_set(train_start_date, 20, 'train_set.csv',all_actions)


# In[ ]:


# 构造测试集
def make_test_set(train_start_date, train_end_date):
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=30)
    start_days = start_days.strftime('%Y-%m-%d')
    all_actions = get_all_action()
    print ("get all actions!")
    user = get_basic_user_feat()
    print ('get_basic_user_feat finsihed')
    product = get_basic_product_feat()
    print ('get_basic_product_feat finsihed')
    
    user_acc = get_recent_user_feat(train_end_date, all_actions)
    print ('get_accumulate_user_feat finsihed')
    
    user_cate = get_user_cate_feature(train_start_date, train_end_date, all_actions)
    print ('get_user_cate_feature finished')
    
    product_acc = get_accumulate_product_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_product_feat finsihed')
    cate_acc = get_accumulate_cate_feat(start_days, train_end_date, all_actions)
    print ('get_accumulate_cate_feat finsihed')
    comment_acc = get_comments_product_feat(train_end_date)

    actions = None
    for i in (3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date, all_actions,i)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date,all_actions,i), how='left',
                               on=['user_id', 'sku_id', 'cate'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    user_cate.index.name = ""
    actions = pd.merge(actions, user_cate, how='left', on='user_id')
    # 注意这里的拼接key
    actions = pd.merge(actions, product, how='left', on=['sku_id', 'cate'])
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, cate_acc, how='left', on='cate')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')

    actions = actions.fillna(0)
    

    actions.to_csv("test_set.csv", index=False)
    


# In[ ]:


make_val_set('2016-02-23', '2016-02-26', 'val_3.csv')


# In[ ]:


#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
from datetime import datetime
import time
from sklearn.model_selection import GridSearchCV


# In[ ]:


data = pd.read_csv('train_set.csv')
data.head()


# In[ ]:


data.columns


# In[ ]:


data_x = data.loc[:,data.columns != 'label']
data_y = data.loc[:,data.columns == 'label']
data_y


# In[ ]:


data_x.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size = 0.2, random_state = 0)


# In[ ]:


x_test.shape


# In[ ]:


x_val = x_test.iloc[:1500,:]
y_val = y_test.iloc[:1500,:]

x_test = x_test.iloc[1500:,:] 
y_test = y_test.iloc[1500:,:]


# In[ ]:


print (x_val.shape)
print (x_test.shape)


# In[ ]:


# 删掉user_id和sku_id两列
del x_train['user_id']
del x_train['sku_id']

del x_val['user_id']
del x_val['sku_id']

x_train.head()


# In[ ]:


dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_val, label=y_val)


# In[ ]:


param = {
    # 树的数量
    'n_estimators': 1000,
    # 树的深度
    'max_depth': 3,
    # 最小叶子权重和
    'min_child_weight': 5,
    'gamma': 0,
    # 用于训练的比例
    'subsample': 1.0,
    # 在建立树时对特征随机采样的比例
    'colsample_bytree': 0.8,
    # 处理类别不均衡
    'scale_pos_weight': 10,
    # 收缩步长
    'eta': 0.1,
    # 学习任务
    'objective': 'binary:logistic',
    # 评价指标
    'eval_metric': 'auc'
}


# In[ ]:


# 迭代次数
num_round = param['n_estimators']

plst = param.items()
evallist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
bst.save_model('bst.model')


# In[ ]:


print (bst.attributes())
# 解释信息


# In[ ]:


# 特征图
def create_feature_map(features):
    outfile = open(r'xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


features = list(x_train.columns[:])
create_feature_map(features)


# In[ ]:


# 特征重要性
def feature_importance(bst_xgb):
    importance = bst_xgb.get_fscore(fmap=r'xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    file_name = 'feature_importance_' + str(datetime.now().date())[5:] + '.csv'
    df.to_csv(file_name)

feature_importance(bst)


# In[ ]:


# 特征重要度
fi = pd.read_csv('feature_importance_10-24.csv')
fi.sort_values("fscore", inplace=True, ascending=False)
fi.head()


# In[ ]:


x_test.head()


# In[ ]:


users = x_test[['user_id', 'sku_id', 'cate']].copy()
del x_test['user_id']
del x_test['sku_id']
x_test_DMatrix = xgb.DMatrix(x_test)
y_pred = bst.predict(x_test_DMatrix, ntree_limit=bst.best_ntree_limit)


# In[ ]:


x_test['pred_label'] = y_pred
x_test.head()


# In[ ]:


def label(column):
    if column['pred_label'] > 0.5:
        #rint ('yes')
        column['pred_label'] = 1
    else:
        column['pred_label'] = 0
    return column
x_test = x_test.apply(label,axis = 1)
x_test.head()        


# In[ ]:


x_test['true_label'] = y_test
x_test.head()


# In[ ]:


x_test['user_id'] = users['user_id']
x_test['sku_id'] = users['sku_id']
x_test.head()


# In[ ]:


# 所有购买用户
all_user_set = x_test[x_test['true_label']==1]['user_id'].unique()
print (len(all_user_set))
# 所有预测购买的用户
all_user_test_set = x_test[x_test['pred_label'] == 1]['user_id'].unique()
print (len(all_user_test_set))
# 用户-商品对
all_user_test_item_pair = x_test[x_test['pred_label'] == 1]['user_id'].map(str) + '-' + x_test[x_test['pred_label'] == 1]['sku_id'].map(str)
all_user_test_item_pair = np.array(all_user_test_item_pair)
print (len(all_user_test_item_pair))


# In[ ]:


pos, neg = 0,0
for user_id in all_user_test_set:
    if user_id in all_user_set:
        pos += 1
    else:
        neg += 1
all_user_acc = 1.0 * pos / ( pos + neg)
all_user_recall = 1.0 * pos / len(all_user_set)
print ('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
print ('所有用户中预测购买用户的召回率' + str(all_user_recall))


# In[ ]:


#所有实际商品对
all_user_item_pair = x_test[x_test['true_label'] == 1]['user_id'].map(
    str) + '-' + x_test[x_test['true_label'] == 1]['sku_id'].map(str)
all_user_item_pair = np.array(all_user_item_pair)
#print (len(all_user_item_pair))
#print(all_user_item_pair)
pos, neg = 0, 0
for user_item_pair in all_user_test_item_pair:
    #print (user_item_pair)
    if user_item_pair in all_user_item_pair:
        pos += 1
    else:
        neg += 1
all_item_acc = 1.0 * pos / (pos + neg)
all_item_recall = 1.0 * pos / len(all_user_item_pair)
print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
print('所有用户中预测购买商品的召回率' + str(all_item_recall))
# 自定义评分标准
F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall +
                                              all_user_acc)
F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall +
                                              3 * all_item_acc)
score = 0.4 * F11 + 0.6 * F12
print('F11=' + str(F11))
print('F12=' + str(F12))
print('score=' + str(score))

