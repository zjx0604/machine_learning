# /usr/local/bin/python3
# _*_ coding:utf-8 _*_
# Author:zhoujinxuan
# @Time:2018-04-02 20:38


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


print("********************导入数据********************")
data = pd.read_csv('/Users/ZJX/PycharmProjects/machine_learning/insurance/finaldata.csv')
X = data[["X2", "X4", "X6", "X9", "X10", "X14", "X15", "X19", "X21", "X23", "X24", "X26", "X27", "X28"]]
y = data[["X36"]].values.ravel()
print(y)
# print(X[:10])
# 查看数据集整体信息
print(data.shape)
# 查看变量缺失值信息
print(data.info())
# 查看前几行数据
print(data.head())
# 统计信息描述
print(data.describe())
# 计算训练集中各个特征值缺失值的总数，无缺失值
print(data.isnull().sum())
# 样本类别均衡
print(data["X36"].value_counts())


print("********************建立模型********************")

print("********************实验A:朴素贝叶斯********************")
# 拆分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 训练模型
gnb = GaussianNB()
gnb.fit(x_train, y_train)
# 测试模型
pred_prob = gnb.predict_proba(x_test)
# 模型评估,计算真正率和假正率(FPR,TPR)
fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:, 1])
# 计算auc的值
roc_auc = auc(fpr, tpr)
# 绘制ROC曲线
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='The AUC of Naive Bayes is %s' % round(roc_auc, 4))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Naive Bayes')
plt.legend(loc="lower right")
plt.show()


print("********************实验B:逻辑斯谛回归********************")
# 拆分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
# 训练模型
logreg = LogisticRegression(class_weight='balanced',
                              tol=0.0001, penalty='l2',
                              solver='liblinear', random_state=0)
logreg.fit(x_train, np.array(y_train.T))
# 测试模型
pred_prob = logreg.predict_proba(x_test)
# 模型评估,计算真正率和假正率
fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:, 1])
# 计算auc的值
roc_auc = auc(fpr, tpr)
# 绘制ROC曲线
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='The AUC of Logistic Regression is %s' % round(roc_auc,4))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Logistic Regression')
plt.legend(loc="lower right")
plt.show()


print("********************实验C:随机森林********************")
# 拆分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 训练模型
random_forest = RandomForestClassifier(max_depth=5,n_estimators=30,random_state=0)
random_forest.fit(x_train, y_train)
# 测试模型
pred_prob = random_forest.predict_proba(x_test)
# 模型评估,计算真正率和假正率
fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:, 1])
# 计算auc的值
roc_auc = auc(fpr, tpr)
# 绘制ROC曲线
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='The AUC of RF is %s' % round(roc_auc,4))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Random Forest')
plt.legend(loc="lower right")
plt.show()


print("********************实验D:AdaBoost********************")
# 拆分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 训练模型
adaboost = AdaBoostClassifier(n_estimators=60,random_state=0)
adaboost.fit(x_train, y_train)
# 测试模型
pred_prob = adaboost.predict_proba(x_test)
# 模型评估,计算真正率和假正率
fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:, 1])
# 计算auc的值
roc_auc = auc(fpr, tpr)
print("The AUC of AdaBoost is :", roc_auc)
# 绘制ROC曲线
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='The AUC of AdaBoost is %s' % round(roc_auc,4))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of AdaBoost')
plt.legend(loc="lower right")
plt.show()