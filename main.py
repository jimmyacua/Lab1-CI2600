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

        print('Magic number: {}\nNumber of images: {}\nRows: {}\nColumns: {}'.format(
            magic,
            images,
            rows,
            columns
        ))

        binary_vector = data.read(images * rows * columns)
        tensor = torch.tensor(list(binary_vector), dtype=torch.uint8)

        tensor = tensor.view(images, rows, columns)

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
    pass


if __name__ == '__main__':
    images = read_idx('train-images.idx3-ubyte')
    labels = read_idx('train-labels.idx1-ubyte')
    print("-----------------------------------------------------------------------------------")
    una = Image.new('L', (28, 28))
    for i in range(0, 5):
        una.putdata(list(images[i].view(-1)))  # el -1 convierte a una dimension
        una.show()
        una.save(str(i)+'.jpg')