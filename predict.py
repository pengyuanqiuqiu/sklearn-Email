# coding:utf-8
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import tree
from getdata import query
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    sql1="select * from  trainmessage limit 0,700" #训练数据
    sql2="select * from  trainmessage limit 699,300"#测试数据
    train=query(sql1)
    test=query(sql2)

    count_v1 = CountVectorizer()
    counts_train = count_v1.fit_transform(train[0])  # 得到词频矩阵

    transformer = TfidfTransformer()  # Transform a count matrix to a normalized tf or tf-idf representation
    tfidf_train = transformer.fit_transform(counts_train)  # 得到频率矩阵

    counts_test = count_v1.transform(test[0])  # fit_transform是将文本转为词频矩阵
    # print(counts_test)
    tfidf_test = transformer.transform(counts_test)  # fit_transform是计算tf-idf
   ####KNN算法
    knnclf = KNeighborsClassifier()
    knnclf.fit(tfidf_train, train[1])
    knn_pred = knnclf.predict(tfidf_test)
    knn_pre = metrics.classification_report(knn_pred, test[1])
    with open('result.txt', 'a+') as save:  ##存入文件中
        save.write('knn model report')
        save.write('\n')
        save.write(knn_pre)
        save.write('\n')
    # print(knn_pre)
    # svm模型
    svc = svm.LinearSVC()
    svc.fit(tfidf_train, train[1])
    svc_pred = svc.predict(tfidf_test)
    svn_pre=metrics.classification_report(svc_pred, test[1])
    # print(svn_pre)
    with open('result.txt','a+') as save:##存入文件中
        save.write('svm model report')
        save.write('\n')
        save.write(svn_pre)
        save.write('\n')
    ###决策树
    DeTree = tree.DecisionTreeClassifier()
    DeTree.fit(tfidf_train, train[1])
    DePre=DeTree.predict(tfidf_test)
    De_pre = metrics.classification_report(DePre, test[1])
    # print(De_pre)
    with open('result.txt', 'a+') as save:  ##存入文件中
        save.write('decision tree model report')
        save.write('\n')
        save.write(De_pre)
        save.write('\n')
    ####随机森林
    # rf = RandomForestRegressor()
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(tfidf_train, train[1])
    rfPre=rf.predict(tfidf_test)
    rfRep=metrics.classification_report(rfPre, test[1])
    with open('result.txt', 'a+') as save:  ##存入文件中
        save.write('random forest model report')
        save.write('\n')
        save.write(rfRep)
        save.write('\n')
    ###bayes 算法
    bayes = MultinomialNB(alpha=0.01)
    bayes.fit(tfidf_train, train[1])
    bayPre=bayes.predict(tfidf_test)
    bayRep=metrics.classification_report(bayPre, test[1])
    with open('result.txt', 'a+') as save:  ##存入文件中
        save.write('bayes model report')
        save.write('\n')
        save.write(bayRep)
        save.write('\n')
    ###logistic regression
    logistic = LogisticRegression(penalty='l2')
    logistic.fit(tfidf_train, train[1])
    logPre=logistic.predict(tfidf_test)
    logRep = metrics.classification_report(logPre, test[1])
    with open('result.txt', 'a+') as save:  ##存入文件中
        save.write('logistic regression model report')
        save.write('\n')
        save.write(logRep)
        save.write('\n')