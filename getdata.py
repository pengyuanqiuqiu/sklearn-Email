import pymysql
import jieba

def query(sql):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='2012213405', db='datamining',
                           charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    result=cursor.fetchall()
    content=[]
    target=[]
    for  res in result:
        temp=[]
        array=jieba.cut(res[2])
        newSet = set()
        for i in array:
            newSet.add(i + ' ')  # 加空格，把不同的词隔开
            # print(i)
            # corpus_train.append(a for a in array)
        newArray = list(newSet)  # 形成字符串列表
        newString = ''.join(newArray)
        # print(newString)
        content.append(newString)
        # print(newString)
        # # for i in str:
        # #     newstr=','.join(i)
        # #     print(newstr)
        # # content.append(newstr)
        target.append(res[1])##y标签  
    return [content,target]




# if __name__ == '__main__':
#     sql="select * from trainmessage"
#     res=query(sql)
#     print(res)