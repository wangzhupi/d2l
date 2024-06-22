import torch 

# 这里的线性模型的任务就是找到最优的参数来进行实现最优化的模型预测

# 损失函数就是模型的度量的东西
import random
from d2l import torch as d2l


# 生成数据
def synthetic_data(w,b, num_examples):
    # 这里其实是模拟的输入而已 数据值在(0,1)之间 shape为(1000,2)
    X = torch.normal(0,1,(num_examples,len(w)))
    print(X.shape)
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    print(y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
print(true_w.shape)
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print("done")
print(f"features is {features[0]}")
print(f"labels is {labels[0]}")
# 改为这种[1000,1]是为了更好的适应机器学习的输出 一般都是[batch_size, num_features]
print(labels.shape)


# 定义一个函数作为读取训练集进行遍历 
def data_iter(batch_size,features,labels):
    # 样本的个数
    num_examples = len(features)
    # 这里是将0-1000和样本排序为数组[0,......,10000]
    indices = list(range(num_examples))
    # 样本随机没有特定的顺序
    #  这里打乱他们的顺序[9,2,0,....876]    
    random.shuffle(indices)
    
    # 这里的循环就是按照批次取得索引
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size,num_examples)]
        )
        # yield 关键字返回一个生成器对象，每次调用时生成一个新的批次
        yield features[batch_indices],labels[batch_indices]
        
batch_size = 10

for X,y in data_iter(batch_size,features,labels):
    print(X, '\n', y)
    break

# 初始化模型参数
# shape为[2,1]的话 就是两行一列 前面为行 后面为列
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# 定义线性回归模型
def linreg(X,w,b):
    
    '''线性回归模型定义'''
    # (num_examples,2) * (2,1) = y(num_examples,1)
    return torch.matmul(X,w) + b

# 定义损失函数
def squared_loss(y_hat,y): #@save
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params,lr,batch_size):
    '''小批量梯度下降'''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 定义需要的参数 基本的模板            
lr = 0.3
num_epochs = 3
net = linreg
loss = squared_loss


for epoch in range(num_epochs):
    # 得到的数据
    for X,y in data_iter(batch_size,features,labels):
        # 损失函数
        l = loss(net(X,w,b),y)
        # 计算梯度
        # 计算出来的梯度存储在w.grad和b.grad当中
        l.sum().backward()
        # 优化参数
        sgd([w,b],lr,batch_size)
        
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')