import tool
import numpy as np
import csv
import time

# 定义几个参数：
b = 1
v = 5
feature_num = 2

all_data = []
with open("feature_2.csv") as csvfile:
    csv_reader = csv.reader(csvfile)                # 使用csv.reader读取csvfile中的文件
    head = next(csv_reader)                         # 读取第一行每一列的标题
    for row in csv_reader:                          # 将csv 文件中的数据保存到all_data中
        for i in range(len(row)):
            row[i] = float(row[i])
        all_data.append(row)

feature_name = head[1:]                             # 特征的名字， 一维list
feature_name = np.array(feature_name)
all_data = np.array(all_data)
label = all_data.T[0]
label = label.tolist()

for i in range(len(label)):
    label[i] = int(label[i])                      # label 一维list

feature =  all_data[:,1:]                         # 所有的特征，二维numpy，
feature = tool.normalize(feature)                 # 把feature 标准化

# num = 0
# for i in range(feature.shape[0]):
#     for j in range(feature.shape[1]):
#         if feature[i][j] != feature[i][j]:
#             num = num +1
# print(num,"个nan")

# 用默认的参数选择特征
best_feature_index = tool.feature_select(feature,label,feature_num,C = 32,gamma = 0.5)
# 用选择出来的特征来选择最佳参数
best_C,best_g,best_score = tool.parameter_select(feature[:,best_feature_index],label)
# 用最佳参数选择最佳特征
best_feature_index = tool.feature_select(feature,label,feature_num,best_C,best_g)

acc = tool.YZ_svm(feature[:,best_feature_index],label,best_C,best_g)
svmopt = ' -c '+str(best_C)+' -g '+str(best_g)+' -b '+str(b)+' -v '+str(v)+' '

result_path='data/train_5_fold_result.txt'
with open(result_path, 'a') as result:
    result.write('-'*6+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                 +'-'*6+'\n')
    result.write('准确率： '
                 +str(acc)+'%\n')
    result.write('选择的特征： '
                 +str(feature_name[best_feature_index])
                 +'\n')
    result.write('svm参数： '
                 +svmopt+'\n')
    result.write("最佳特征下标： ")
    for i in range(len(best_feature_index)):
        result.write(str(best_feature_index[i])+" ")

print("Best feature is :",feature_name[best_feature_index])
print("parameter is：",svmopt)
print("acc:",acc)
