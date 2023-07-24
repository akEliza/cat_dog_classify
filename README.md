# cat_dog_classify
it's a simple project based on Pytorch which can classify whether an image is a cat or a dog

## 超参数选择

一共训练了30个epoch，batchsize设定为100。

使用Adam算法进行优化，learning_rate选择为1e-3。

loss_function选择为交叉熵损失函数。

## 结果



部分epoch训练结果如下(训练平台autodl，RTX3090）：

## ![image1](https://github.com/akEliza/cat_dog_classify/blob/master/image-202307111091625734.png)

最终测试集上测试出的结果（在本机上跑的，一块GTX1650）：

![image-20230711102417459](https://github.com/akEliza/cat_dog_classify/blob/master/image-2023071102417459.jpg)

Training和Validation的loss曲线![loss](https://github.com/akEliza/cat_dog_classify/blob/master/loss.png)

Training和Validation的Accuracy曲线

![accuracy](https://github.com/akEliza/cat_dog_classify/blob/master/accuracy.png)



## divide_data.py

把原始猫狗数据集划分为训练集、验证集和测试集，比例为6:2:2。

## ResNet18.py

搭建用于图片分类的resnet18网络

## Train.py

训练模型代码。模型训练的参数存储在params.pth中。

## Pred.py

测试模型在测试集上的准确率代码。
