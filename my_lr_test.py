#logit regression model
import numpy as np
import matplotlib.pyplot as plt
colors = np.array(["g", "b", "r", "m"])

def generator_dots(class_num, dots_num):
    # tags = np.random.choice([-1, 1], size=[dots_num])
    x = np.random.random(size=[dots_num]) * 10
    y = np.random.random(size=[dots_num]) * 10
    # y = x + np.random.normal(2, 1, size=[dots_num]) * tags
    dots = np.vstack((x, y)).T
    tags = np.zeros([dots_num], dtype=int)
    for i in np.arange(dots_num):
        gap = dots[i, 1] - dots[i, 0]
        weight = np.array([np.exp(gap), np.exp(-gap)])
        weight = weight/weight.sum()
        tags[i] = np.random.choice([0, 1], p=weight)
    return dots, tags

def draw_scatter(dots, tags):
    plt.figure(0)
    dots_colors = colors[tags]
    for dot, color in zip(dots, dots_colors):
        # print(dot, color)
        plt.scatter(dot[0], dot[1], color=color)
    plt.xlim([0, 10])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("generatted dots")

def draw_plot(weight, bias):
    weight = np.reshape(weight, newshape=[-1])
    dim = weight.shape[0]
    if dim != 2:
        print("版本1仅支持画二维图像")
        return
    a, b = weight
    x = np.arange(0, 10, 0.1)
    y = (-a * x - bias)/b
    plt.figure(0)
    plt.plot(x, y)

class LR:
    def __init__(self, dots=None, tags=None):
        self.weight = None
        self.bias = None
        if dots is not None and tags is not None:
            self.train_model(dots, tags)

    def train_model(self, dots, tags):
        dataNum, dim = dots.shape
        if self.weight is None:
            self.weight = np.random.normal(0, 1, size=[dim, 1])
        if self.bias is None:
            self.bias = 0
        # 开始训练
        maxIter = 10000
        old_loss = 0
        tags = np.reshape(tags, newshape=[dataNum, 1])
        alpha = 0.01
        for i in np.arange(maxIter):
            y_hat = 1.0/(1 + np.exp(dots.dot(self.weight) + self.bias))
            loss = -(tags.T.dot(np.log(y_hat)) + (1 - tags).T.dot(np.log(1 - y_hat)))
            error = y_hat - tags
            grad = dots.T.dot(error)/dots_num
            self.weight = self.weight + grad * alpha
            self.bias = self.bias + np.mean(error) * alpha
            if 0 < old_loss - loss <= 0.01:
                print("step %d 训练结束, 损失为 %.4f" % (i, loss))
                break
            old_loss = loss
            if i % 10 == 0:
                print("step %d, loss: %.4f" % (i, loss))


    def get_para(self):
        return self.weight, self.bias


if __name__ == "__main__":
    dots_num = 200
    class_num = 2
    dots, tags = generator_dots(class_num=class_num, dots_num=dots_num)
    # print(dots.shape, tags.shape)
    draw_scatter(dots, tags)
    model = LR()
    model.train_model(dots, tags)
    weight, bias = model.get_para()
    draw_plot(weight, bias)

    plt.show()
