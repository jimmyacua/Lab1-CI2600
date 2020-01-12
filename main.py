import torch
import struct as st
from PIL import Image
import os
import random


def read_idx(archivo):
    data = open(archivo, 'rb')

    # magic_number = int.from_bytes(data.read(4), byteorder="big", signed=True)
    data.seek(0)
    magic = st.unpack('>4B', data.read(4))
    # print(magic[3])

    if magic[3] == 3:
        images = int.from_bytes(data.read(4), byteorder="big", signed=True)

        rows = int.from_bytes(data.read(4), byteorder="big", signed=True)

        columns = int.from_bytes(data.read(4), byteorder="big", signed=True)

        '''print('Magic number: {}\nNumber of images: {}\nRows: {}\nColumns: {}'.format(
            magic,
            images,
            rows,
            columns
        ))'''

        binary_vector = data.read(images * rows * columns)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        tensor = tensor.view(images, rows, columns)
        # print(tensor)
        return tensor

    elif magic[3] == 1:
        labels = int.from_bytes(data.read(4), byteorder="big", signed=True)

        print('Magic number: {}\nNumber of labels: {}'.format(
            magic,
            labels,
        ))

        binary_vector = data.read(labels)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        print("LABELS", tensor.view(labels))

        return tensor


def save_images(images):
    una = Image.new('L', (28, 28))
    for i in range(0, 5):
        una.putdata(list(images[i].view(-1)))  # el -1 convierte a una dimension
        una.show()
        una.save(str(i) + '.jpg')
        

def filter_data(images, labels, singleLabel):
    x = (labels == singleLabel)
    y = x.nonzero()
    nums = images[y]
    image = Image.new('L', (28, 28))
    image.putdata(list(nums[random.randint(0, nums.size()[0])].view(-1)))
    #image.show()
    image.save(os.path.join('./filter_data/'+str(singleLabel) + '.jpg'))
    return nums


def merge_images(images, operation, number):
    labels = read_idx('train-labels.idx1-ubyte')
    x = (labels == number)
    y = x.nonzero()
    if operation == "max":
        nums = images[y]
        max = torch.max(nums)
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[max.type(torch.int32)].view(-1)))
        #ximage.show()
        image.save(os.path.join('./max/'+str(number) + '.jpg'))
    elif operation == "median":
        nums = images[y]
        media = torch.median(nums)
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[media.type(torch.int32)].view(-1)))
        #image.show()
        image.save(os.path.join('./median/'+str(number) + '.jpg'))
    elif operation == "mean":
        nums = images[y]
        mean = torch.mean(nums.type(torch.float32))
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[mean.type(torch.int32)].view(-1)))
        #image.show()
        image.save(os.path.join('./mean/'+str(number) + '.jpg'))
    else:
        print("Error! Operation not found")


if __name__ == '__main__':
    images = read_idx('train-images.idx3-ubyte')
    labels = read_idx('train-labels.idx1-ubyte')
    '''for i in range(0, 10):
        filter_data(images, labels, i)'''
    '''for i in range(0, 10):
        merge_images(images, "max", i)
        merge_images(images, "median", i)
        merge_images(images, "mean", i)'''
    #merge_images(images, "median", 1)
    filter_data(images, labels, 7)
