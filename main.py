from typing import List

from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


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

def conv(pic,kernel_number,stride,pad):

    width = len(pic[0])
    height = len(pic)
    print(width,height)
    picdata = [[0 for i in range(width)] for j in range(height)]
    kernelset = []
    for y in range(height):
        for x in range(width):
            picdata[y][x] = (sum(pic[y][x])//3)
    for i in range(kernel_number):
        kernelset.append([[random.randrange(-1,2,2),random.randrange(-1,2,2),random.randrange(-1,2,2)],
                [random.randrange(-1,2,2),random.randrange(-1,2,2),random.randrange(-1,2,2)],
                [random.randrange(-1,2,2),random.randrange(-1,2,2),random.randrange(-1,2,2)]])

    output_width = int((width-2)//stride+1)
    output_height = int((height-2)//stride+1)
    print(output_width)
    print(output_height)
    final_data = [[[0 for u in range(len(kernelset))] for i in range(output_width)] for k in range(output_height)] #(iutput+2*p-3)/s+1
    print(final_data)
    for k in range(len(kernelset)):
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
    # print("conv")
    # for i in range(len(final_data)):
    #     print(final_data[i])
    # print("_______")
    print(len(final_data))
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
    # print("_______")
    print(len(tmp_data),len(tmp_data[0]))

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
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.multiply(sigmoid(x),(1-sigmoid(x)))

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

def backprop(input,ans,output,stock,learn,w_1,w_2):
    #loss-W1 loss_derivative = X'(sig'(z2)w'2)sig'(z1)
    input = np.array(input)
    z1 = np.array(stock[1]).reshape(1, -1)  # 確保形狀是 (1, 400)
    z2 = np.array(stock[2]).reshape(1, -1)  # 確保形狀是 (1, 3)
    compute_gradients_1 = np.dot(sigmoid_derivative(z2),np.transpose(w_2))
    compute_gradients_1 = np.multiply(compute_gradients_1,sigmoid_derivative(z1))
    compute_gradients_1 = np.dot(np.transpose(input.reshape(1, -1)),compute_gradients_1)
    w_1 = w_1-compute_gradients_1*learn
    compute_gradients_2 = ((sigmoid(z2)-ans)*sigmoid_derivative(z2))
    a1_T = np.transpose(sigmoid(z1))
    compute_gradients_2 = np.dot(a1_T,compute_gradients_2)
    w_2 = w_2-compute_gradients_2*learn
    return w_1,w_2

if __name__ == '__main__':
    tmp = importImgToData("/Users/mason/Documents/Cnn/train.jpg")
    #testimg = Image.fromarray(np.uint8(tmp))
    #estimg.show()
    answer = [1,0,0]
    convOutput = conv2D(tmp,8,2)
    produce = conv2D(convOutput,8,2)
    process_data = flatten(produce)
    LR = 0.0001
    fla_size = len(process_data)
    print(fla_size)
    Warrant_1 = np.random.uniform(-0.1, 0.1, size=(fla_size, 400))
    Warrant_2 = np.random.uniform(-0.1, 0.1, size=(400, 3))
    time = 0

    for i in range(10000):
        time +=1
        forward,z_stock = forwardprop(process_data,Warrant_1,Warrant_2)
        Warrant_1,Warrant_2 = backprop(process_data,answer,forward,z_stock,LR,Warrant_1,Warrant_2)
        if time == 100:
            loss = 0
            for f in range(3):
                loss += (answer[f]-forward[f])**2
            time = 0
            print(loss/fla_size)




