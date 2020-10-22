import matplotlib.pyplot as plt
import os
import numpy as np

def o2svm(feature, label, trainsvm_txt_path):
    """
    该函数的功能是数据转换，将原来的数据转化成exe识别的格式
    :param feature: 需要训练的特征，二维numpy
    :param label: 标签
    :param trainsvm_txt_path: 写入的文档
    :return:
    """
    [nr, nc] = feature.shape
    with open(trainsvm_txt_path, 'w') as sf:               # 追加写，‘w’覆盖写
        for i in range(nr):
            sf.write(str(label[i]) + ' ')                  # 标签label
            for j in range(nc):
                sf.write(str(j + 1) + ':' + str(feature[i][j]) + ' ')
            sf.write('\n')

def CalDXmulti(train_feature,train_label):
    """
    计算DXscore
    :param train_feature: 有多少特征就多少列，所有的标签下的数据形成行
    :param train_label:   是一个列向量 都要numpy型
    :return:
    """
    featnum = train_feature.shape[1]                 #特征数
    print("featnum",featnum)
    DX = np.zeros(featnum)
    labelnum = np.unique(train_label).ravel()
    K = len(labelnum)                                # 分类数
    print("this is a {} problem".format(K))
    features = []

    for i in range(K):
        features.append(train_feature[np.where(train_label==labelnum[i])])

    for i in range(featnum):
        m = np.zeros(K)
        v = np.zeros(K)
        for j in range(K):
            ft = features[j][:,i]
            m[j] = np.mean(ft)
            v[j] = np.var(ft)
        iDX = 0
        for k in range(K):
            for l in range(k+1,K):
                mkl = (m[k] - m[l])*(m[k] - m[l])
                vkl =  v[k] + v[l]
                if vkl > 0:
                    iDX = iDX + mkl/vkl
                elif mkl > 0:
                    iDX +=1
        DX[i] = iDX

    index = np.argsort(-DX)
    return index

def YZ_svm(feature,label,c= 32,gamma = 0.5,b = 1,v = 5):
    """
    此函数的功能是对于输入的特征进行v=5折交叉验证，并返回测试结果
    :param feature: 进行学习的特征
    :param label: 一维标签数组
    :param c:
    :param gamma:
    :param b:   返回的不是预测的标签，而是每个标签的概率
    :param v:   v折交叉验证
    :return:    测试集的准确率
    """

    # ----格式转换----
    trainsvm_txt_path = 'data/trainsvm.txt'        # 储存训练得到的SVM参数
    o2svm(feature, label, trainsvm_txt_path)
    # ----调用svm_train.exe----
    svm_train_exe_path = 'svm-train.exe'
    svmlog_path = 'data/svmlog.txt'
    svmopt = ' -c ' + str(c) + ' -g ' + str(gamma) + ' -b ' + str(b) + ' -v ' + str(v) + ' '

    os.system(svm_train_exe_path + svmopt + ' ' + trainsvm_txt_path + '>' + svmlog_path)

    # -----从svmlog文件中读入acc----
    acc = 0
    data = []
    for line in open(svmlog_path):
        data.append(line)
    line = data[-1]
    istart = np.array([j for j in range(len(line)) if line.startswith('Cross Validation Accuracy = ', j)]) + len(
        'Cross Validation Accuracy = ')
    if len(istart) == 0:
        print('Accuracy not find!')
        return 0
    elif len(istart) != 0:
        iend = ([j for j in range(len(line)) if line.startswith('%', j)])
        acc = float(line[istart[0]:iend[0]])
    return acc


def normalize(feature,method = 1):
    """
    归一化,但是现在有问题，对于Max == Min的情况，无法解决
    :param feature: 原数据
    :param method:  等于0 则归一化到0~1
                    等于1 则归一化到-1~1
                    default：减去均值除以方差
    :return:
    """
    Max = feature.max ( axis = 0 )
    Min = feature.min ( axis = 0 )
    Mean= feature.mean( axis = 0 )
    Var = feature.var ( axis = 0 )
    if method == 1:
        feature = 2*(feature - Mean)/(Max - Min)
    elif method == 0:
        feature = (feature - Min)/(Max - Min)
    else:
        feature = (feature -Mean)/Var
    return feature

def feature_select(features,labels,maxnumf,C = 32,gamma = 0.5,b = 1,v = 5):
    """
    :param features: 是一个二维数组numpy
    :param labels: 是一组特征下的标签
    :param maxnumf: 是需要选择的特征数
    :param C: svm的参数
    :param gamma: svm的参数
    :param b: b == 1 则返回的是概率
    :param v: 几折交叉验证，默认为5
    :return: 返回最佳参数的下标
    """

    if features.shape[1] < maxnumf :
        return("len(features) is less than maxnumf")

    featnum = min(features.shape[1],maxnumf)
    DX = CalDXmulti(features,labels)

    score = []
    for i in range(featnum):
        n_feat = DX[:i+1]                                       #n_feat是标签索引的数组
        trainfeat = features[:,n_feat]
        score.append(YZ_svm(trainfeat,labels,C ,gamma ,b ,v ))

    X = range(1,featnum+1)
    plt.plot(X,score)
    plt.scatter(score.index(max(score))+1,max(score))
    plt.annotate("({},{})".format(score.index(max(score)) + 1, max(score)),
                 xy=(score.index(max(score)) + 1, max(score)))
    plt.show()

    return DX[:score.index(max(score))+1]

def parameter_select(feature,label):
    C_group = []
    gamma_group = []
    for i in range(-1,15,2):
        C_group.append(2**i)
    for i in range(7,-15,-2):
        gamma_group.append(2**i)
    score = []
    for C in C_group:
        for gamma in gamma_group:
            score.append(YZ_svm(feature,label,C,gamma))
            print([C,gamma,score])
    best_score = max(score)
    index = score.index(best_score)
    best_C = C_group[index//len(gamma_group)]
    best_g = gamma_group[index%len(gamma_group)]
    return best_C,best_g,best_score

