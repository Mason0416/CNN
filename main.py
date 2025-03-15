from typing import List

from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
SHARED_KERNELSET = None  # 全局變數


# 載入 MNIST 資料
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

# 將灰階圖轉成 RGB-like 結構（因為你的 conv 是針對 RGB 的）
def expand_gray_to_rgb(img):
    return [[[int(p)/256] * 3 for p in row] for row in img]

def preprocess_mnist_images(x_data, y_data, limit=100):
    processed_data = []
    count = 0
    for i in range(len(x_data)):
        if y_data[i] not in [3, 4, 5]:
            continue
        rgb_img = expand_gray_to_rgb(x_data[i])
        conv_out = conv2D(rgb_img, 16, 2)
        conv_out = conv2D(conv_out, 16, 2)
        flat = flatten(conv_out)
        label = one_hot(y_data[i])
        processed_data.append((flat, label))
        count += 1
        if count == limit:
            break
    return processed_data


def importImgToData(path):

    orgImg = Image.open(path)
    pxMatrix = np.asarray(orgImg)
    #print("pic dim :", pxMatrix.ndim)

    height = orgImg.height
    width = orgImg.width    
    data = [[[0 for rgb in range(3)] for x in range(width)] for y in range(height)]
    #print(height, width)

    for y in range(height):
        for x in range(width):
            r, g, b = pxMatrix[y][x]
            #print(r, g, b)
            data[y][x][0] = r
            data[y][x][1] = g
            data[y][x][2] = b
    # print("image")
    # for i in range(len(data)):
    #     print(data[i])
    # print("_______")
    return data

def init_shared_kernel(kernel_number):
    global SHARED_KERNELSET
    SHARED_KERNELSET = []
    for i in range(kernel_number):
        kernel = [[random.randrange(-1, 2, 2) for _ in range(3)] for _ in range(3)]
        SHARED_KERNELSET.append(kernel)

def conv(pic,kernel_number,stride,pad):
    global SHARED_KERNELSET
    width = len(pic[0])
    height = len(pic)
    print(width,height)
    picdata = [[0 for i in range(width)] for j in range(height)]
    
    for y in range(height):
        for x in range(width):
            picdata[y][x] = int(sum(pic[y][x]) // 3)
    
    output_width = int((width-2)//stride+1)
    output_height = int((height-2)//stride+1)
    
    final_data = [[[0 for u in range(len(SHARED_KERNELSET))] for i in range(output_width)] for k in range(output_height)] #(iutput+2*p-3)/s+1
    
    for k in range(len(SHARED_KERNELSET)):
        kernelset = SHARED_KERNELSET
        for y in range(1,height-1,stride):
            for x in range(1,width-1,stride):

                con = 0
                con += picdata[y-1][x-1] * kernelset[k][0][0]
                con += picdata[y-1][x] * kernelset[k][0][1]
                con += picdata[y-1][x+1] * kernelset[k][0][2]
                con += picdata[y][x - 1] * kernelset[k][1][0]
                con += picdata[y][x] * kernelset[k][1][1]
                con += picdata[y][x + 1] * kernelset[k][1][2]
                con += picdata[y + 1][x - 1] * kernelset[k][2][0]
                con += picdata[y + 1][x] * kernelset[k][2][1]
                con += picdata[y + 1][x + 1] * kernelset[k][2][2]

                final_data[y//stride][x//stride][k] = con
    
    return final_data

def pooling(feature):
    width = len(feature[0])
    height = len(feature)
    final_data = [[[0 for j in range(len(feature[0][0]))] for i in range(width-2)] for k in range(height-2)]

    for k in range(len(feature[0][0])):
        for y in range(1,height-1):
            for x in range(1,width-1):

                tmp = []
                tmp.append(feature[y-1][x-1][k])
                tmp.append(feature[y-1][x][k])
                tmp.append(feature[y-1][x+1][k])
                tmp.append(feature[y][x - 1][k])
                tmp.append(feature[y][x][k])
                tmp.append(feature[y][x + 1][k])
                tmp.append(feature[y + 1][x - 1][k])
                tmp.append(feature[y + 1][x][k])
                tmp.append(feature[y + 1][x + 1][k])
                final_data[y-1][x-1][k] = (max(tmp))
    # print("pooling")
    # for i in range(len(final_data)):
    #     print(final_data[i])

    return  final_data


def relu(pic):
    y = len(pic)
    x = len(pic[0])
    z = len(pic[0][0])
    for i in range(y):
        for u in range(x):
            for k in range(z):
                if pic[i][u][k] < 0:
                    pic[i][u][k] = 0
    return pic
def padding(pic,time):
    tmp_data = [[[0 for i in range(len(pic[0][0]))]for u in range(len(pic[0]) + time)]for k in range(len(pic)+time)]
    for y in range(len(pic)):
        for x in range(len(pic[0])):
            tmp_data[y][x] = pic[y][x]
    # print("padding")
    # for i in range(len(tmp_data)):
    #     print(tmp_data[i])
    # print("_______")polacode.activate
    

    return tmp_data

def flatten(data):
    y = len(data)

    x = len(data[0])
    z = len(data[0][0])
    final = []
    for i in range(y):
        for u in range(x):
            for a in range(z):
                final.append(data[i][u][a])
    final = np.array(final).reshape(1,-1)
    return final

def conv2D(pic,kernel_amount,stride):
    tmp = (len(pic)-1)%stride
    if tmp == 1:
        process_data = padding(pic,1)
    else:
        process_data = pic
    process_data = conv(process_data,kernel_amount,stride,tmp)
    process_data = pooling(process_data)
    process_data = relu(process_data)

    return process_data


def sigmoid(x):
    x = np.clip(x,-500,500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):

    return np.multiply(sigmoid(x),(1-sigmoid(x)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 避免數值爆炸
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def hidden_layer(data,w):
    #fla_size = len(data)
    #W1 = np.random.uniform(-0.1,0.1,size=(fla_size,400))
    return np.dot(data,w)

def predict(data,w):
    #W2 = np.random.uniform(-0.1,0.1,size=(400,3))
    return np.dot(data,w)

def forwardprop(data,w_1,w_2):
    z = [0]
    hid = hidden_layer(data,w_1)
    z.append(hid)
    hid = sigmoid(hid)
    pre = predict(hid,w_2)
    z.append(pre)
    pre = sigmoid(pre)

    return pre,z

def backprop(conved_data,ans,output,stock,learn,w_1,w_2):
    #loss-W1 loss_derivative = X'(sig'(z2)w'2)sig'(z1)
    #conved_data = np.array(conved_data)
    z1 = np.array(stock[1]).reshape(1, -1)  # 確保形狀是 (1, 400)
    z2 = np.array(stock[2]).reshape(1, -1)  # 確保形狀是 (1, 10)
    delta_output = (output - ans)*sigmoid_derivative(z2)# (1, 10)
    delta_hidden = np.dot(delta_output, w_2.T) * sigmoid_derivative(z1)  # (1, 400)
    compute_gradients_1 = np.dot(conved_data.reshape(-1, 1), delta_hidden)  # (input, hidden)
    w_1 = w_1-compute_gradients_1*learn
    compute_gradients_2 = (output-ans)
    a1_T = np.transpose(sigmoid(z1))
    compute_gradients_2 = np.dot(a1_T,compute_gradients_2)
    w_2 = w_2-compute_gradients_2*learn
    return w_1,w_2

def one_hot(label):
    vec = [0]*3
    vec[label-3] = 1
    return vec


if __name__ == '__main__':
    init_shared_kernel(8)  # 假設 conv2D 用的是 8 個 kernel
    # 前處理前 100 筆訓練資料
    train_data = preprocess_mnist_images(x_train_raw, y_train_raw, limit=5000)
    # 處理測試資料（可限制幾張）
    test_data = preprocess_mnist_images(x_test_raw, y_test_raw, limit=100)
    LR = 0.0001
    fla_size = train_data[0][0].shape[1]
    print(fla_size)
    Warrant_1 = np.random.uniform(-0.1, 0.1, size=(fla_size, 400))
    Warrant_2 = np.random.uniform(-0.1, 0.1, size=(400, 3))
    epochs = 200000
    loss_list = []
    for epoch in range(epochs):
        print((epoch+1))
        random.shuffle(train_data)
        for i,(x,y) in enumerate(train_data):
            forward,z_stock = forwardprop(x,Warrant_1,Warrant_2)
            Warrant_1,Warrant_2 = backprop(x,y,forward,z_stock,LR,Warrant_1,Warrant_2)
            if i %100 == 0:
                loss = np.sum((np.array(y) - forward)**2) / 2  
                print(f"[{i}] Loss: {loss:.4f}")
                loss_list.append(loss)
                if loss <= 0.1:
                    break

    

    # 測試準確率
    correct = 0
    total = len(test_data)

    for x, y in test_data:
        output, _ = forwardprop(x, Warrant_1, Warrant_2)
        predicted = np.argmax(output)
        actual = np.argmax(y)
        if predicted == actual:
            correct += 1

    accuracy = correct / total
    print(f"\n✅ 測試集準確率：{accuracy * 100:.2f}% ({correct}/{total})")

    plt.plot(loss_list, label='Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Step (每10筆記一次)')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()