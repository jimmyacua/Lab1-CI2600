import torch
import struct as st
from PIL import Image


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
        

def filter_data(images, labels, singleLabel, index_to_save, name):
    x = (labels == singleLabel)
    y = x.nonzero()
    nums = images[y]
    image = Image.new('L', (28, 28))
    image.putdata(list(nums[index_to_save].view(-1)))
    #image.show()
    image.save(name + '.jpg')
    return nums


def merge_images(images, operation, number):
    labels = read_idx('train-labels.idx1-ubyte')
    if operation == "max":
        max = torch.max(images).item()
        x = (labels == number)
        y = x.nonzero()
        nums = images[y]
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[max].view(-1)))
        image.show()
    elif operation == "median":
        media = torch.median(images).item()
        x = (labels == number)
        y = x.nonzero()
        nums = images[y]
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[media].view(-1)))
        image.show()
    elif operation == "mean":
        x = (labels == number)
        y = x.nonzero()
        nums = images[y]
        mean = torch.mean(nums, 1)
        image = Image.new('L', (28, 28))
        image.putdata(list(nums[mean].view(-1)))
        image.show()


if __name__ == '__main__':
    images = read_idx('train-images.idx3-ubyte')
    '''labels = read_idx('train-labels.idx1-ubyte')
    print("-----------------------------------------------------------------------------------")
    #save_images(images)
    filter_data(images, labels, 0, 19, "zero")
    filter_data(images, labels, 1, 87, "one")
    filter_data(images, labels, 2, 32, "two")
    filter_data(images, labels, 3, 55, "three")
    filter_data(images, labels, 4, 23, "four")
    filter_data(images, labels, 5, 1, "five")
    filter_data(images, labels, 6, 45, "six")
    filter_data(images, labels, 7, 12, "seven")
    filter_data(images, labels, 8, 56, "eight")
    filter_data(images, labels, 9, 43, "nine")'''
    merge_images(images, "mean", 4)
