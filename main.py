import xlrd #要下载1.2.0版本，否则不能读xlsx文件
from math import log2
from random import shuffle
from sklearn.metrics import classification_report

def Ent(data):#信息熵
    label={}
    for i in data:
        if i[-1] not in label.keys():
            label[i[-1]]=0
        label[i[-1]] += 1
    ans=0.0
    for key in label:
        p=float(label.get(key))/len(data)
        ans-=p*log2(p)#信息熵的计算
    return ans

def Gain(data,a):#计算属性a对样本集进行划分所获得的信息增益
    ent=Ent(data)
    feature=[]
    for i in data:
        feature.append(i[a])
    feature=set(feature)
    for f in feature:
        num=[]
        for i in data:
            if i[a]==f:
                num.append(i)
        p=len(num)/len(data)
        ent-=p*Ent(num)
    return ent

def most_in_label(label):
    cnt={}
    for i in label:
        if i not in cnt:
            cnt[i]=0
        cnt[i]+=1
    tmp=0
    index=0
    for i in cnt.keys():
        if cnt.get(i)>tmp:
            tmp,index=cnt.get(i),i
    return index

#离散值处理
def Discrete_value_processing(data):
    best_ent, best_fea = -1e9, 0
    for i in range(len(data[0]) - 1):
        if type(data[0][i]).__name__ == 'float': #如果数据是连续值
            continue
        if Gain(data, i) > best_ent:
            best_ent, best_fea = Gain(data, i), i
    return best_ent, best_fea

#连续值处理
def Continuous(data,i):
    candi, l = [], [d[i] for d in data]
    l = list(set(l))  # 需要不同的取值，所以需要去重
    for pos in range(1, len(l)):  # 上下命名不能重复
        candi.append((l[pos] + l[pos - 1]) / 2)  # 取中位数
    max_ent, best_candi, base_ent = -1e9, 0, Ent(data)
    for can in candi:
        # 以这个值作为划分点
        l_data, r_data = [], []
        # 获得左侧右侧元素
        for d in data:
            if d[i] > can:
                r_data.append(d)
            else:
                l_data.append(d)
        # 获取信息熵和比例
        ent1, posi1 = Ent(l_data), len(l_data) / len(data)
        ent2, posi2 = Ent(r_data), len(r_data) / len(data)
        tmp_ent = base_ent - posi1 * ent1 - posi2 * ent2
        # 更新
        if tmp_ent >= max_ent:
            max_ent = tmp_ent
            best_candi = can
    # 返回最佳信息增熵以及划分点
    return max_ent, best_candi


def Continuous_value_processing(data):
    best_ent, best_fea, best_val = -1e9, 0, 0
    for i in range(len(data[0]) - 1):
        if type(data[0][i]).__name__ != 'float': #如果数据是离散值
            continue
        candi_ent, candi_val = Continuous(data, i)
        # 如果以这个属性作为划分，找到的最佳划分点，划分出来的信息增熵最大，那么选择这个
        if candi_ent > best_ent:
            best_ent, best_fea, best_val = candi_ent, i, candi_val
    return best_ent, best_fea, best_val

def dfs(data,name):
    label=[]
    for i in data:
        label.append(i[-1])
    if label.count(label[0])== len(label):
        return label[0] #如果只有一个类型，那么直接返回这个类型
    if len(data)==0:
        return most_in_label(label) #如果数据集为空，将分支节点标记为叶子结点，类别标记为D中样本最多的类(西瓜书原话)
    max_ent, best = Discrete_value_processing(data)
    max_ent2, best2, best2_val = Continuous_value_processing(data)
    if max_ent2 < max_ent:  # 如果是离散的
        best_label = name[best]
        Tree = {best_label: {}}
        del (name[best])
        dif_fea = set([i[best] for i in data])
        for fea in dif_fea:
            n_data, n_name = [], name[:]
            for i in data:
                if (i[best] == fea):
                    tmp = []
                    for j in range(len(i)):
                        if j != best: tmp.append(i[j])
                    n_data.append(tmp)
            Tree[best_label][fea] = dfs(n_data, n_name)
        return Tree
    else:  # 如果是连续的的
        best_label = name[best2]
        Tree = {best_label: {}}
        # 小于，大于，分成两段
        n_data, n_name, n_data2, n_name2 = [], name[:], [], name[:]
        for i in data:
            if i[best2] < best2_val:
                n_data.append(i)
            else:
                n_data2.append(i)
        # dfs递归，分成小于和大于等于
        Tree[best_label]["<" + str(best2_val)] = dfs(n_data, n_name)
        Tree[best_label][">=" + str(best2_val)] = dfs(n_data2, n_name2)
        return Tree

def read_watermelon():
    path="water.xlsx"
    book = xlrd.open_workbook(path)
    sheet = book.sheets()[0]
    row, col = sheet.nrows, sheet.ncols
    data, label = [], []
    for i in range(1, row):
        tmp = []
        for j in range(1, col):
            tmp.append(sheet.cell(i, j).value)
        data.append(tmp)
    tmp = []
    for j in range(1, col): tmp.append(sheet.cell(0, j).value)
    label = tmp
    return data, label


data,label=read_watermelon()
shuffle(data) #打乱数据集，使用4/5用于训练
train,test=[],[]
for i in range(len(data)):
    if i<len(data)*0.8:
        train.append(data[i])
    else:
        test.append(data[i])
Tree = dfs(train,label[:])#获取到剪枝前的树
test_y=[i[-1] for i in test]
pred_y=[]


def predict(Tree, name, test):
    # 获取目前属性划分
    feature = list(Tree.keys())[0]
    dic2 = Tree[feature]
    index = name.index(feature)
    # 如果是连续属性
    if (type(test[index]).__name__ == 'float'):
        val = list(dic2.keys())[0]
        val = float(val[1:])
        if (test[index] < val):
            res = '<' + str(val)
        else:
            res = '>=' + str(val)
        # 是否走到了叶子节点
        if (type(dic2[res]).__name__ == 'dict'):
            return predict(dic2[res], name, test)
        else:  # 走到叶子节点，返回结果
            return dic2[res]
    for key in dic2.keys():
        if test[index] == key:  # 没有走到叶子节点，继续递归
            if (type(dic2[key]).__name__ == 'dict'):
                return predict(dic2[key], name, test)
            else:  # 走到叶子节点
                return dic2[key]

def cut(root,Tree,fa,fa_key,name,test,label):
    #获取到目前键值的名字
    feature=list(Tree.keys())[0]
    dic2 = Tree[feature]
    #获取到对应的下标
    index=name.index(feature)
    #获取到所有子节点
    for key in dic2.keys():
        #如果子节点是字典，那么尝试对它进行剪枝
        if (type(dic2[key]).__name__ == 'dict'):
           cut(root,dic2[key],Tree,key,name,test,label)
    #需要存储原始的结果，防止没有剪枝后没有更优，需要回退
    Tree2 = Tree.copy()
    best_acc, best_label = Accuracy(root, name, test), ''
    base_acc = best_acc
    #根节点不能剪枝
    if(Tree==root):return
    for i in label:
        #由于不传data，所以不知道哪个label最多，选择遍历所有label找最好的。复杂度为|S|*节点数，s为不同的分类结果个数
        fa[list(fa.keys())[0]][fa_key]=i
        #准确率是否有提高
        now_acc = Accuracy(root, name, test)
        if (now_acc > best_acc):
            best_acc, best_label = now_acc, i
    if (best_acc > base_acc):
        fa[list(fa.keys())[0]][fa_key]=best_label
    else:
        fa[list(fa.keys())[0]][fa_key]=Tree2
# 建树


def Accuracy(Tree, label, test):
    cnt = 0
    for d in test:  # 获取准确率
        if (predict(Tree, label, d[:-1]) == d[-1]):
            cnt += 1
    return cnt / len(test)


print('剪枝前准确率：',Accuracy(Tree,label,test))
for i in test:
    pred_y.append(predict(Tree,label,i))
print('剪枝前决策树报告')
print(classification_report(test_y,pred_y) )#查看report
print(classification_report(test_y,pred_y) )#查看report
cut(Tree,Tree,0,0,label,test,set([i[-1] for i in test]))
print('剪枝后准确率：',Accuracy(Tree,label,test))
pred_y=[]
for i in test:
    pred_y.append(predict(Tree,label,i))
print('剪枝后决策树报告')
print(classification_report(test_y,pred_y) )#查看report