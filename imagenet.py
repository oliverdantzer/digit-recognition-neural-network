import numpy as np
import pygame as pg
import csv

# Each layer is matrix of size a X b, where b is the # of inputs, and a is the size of the next layer

# given a layer i's activation a^i, a^(i+1) = sigmoid(W^i * a^i + b) where W is weight matrix for layer i and b is the biases vector for layer i

# Let's say our neural network takes a 28x28 image, and it outputs a digit from 0-9 that it thinks the image is

class Image:
    def __init__(self, label: int, pixels: list[int]) -> None:
        self.label = label
        self.pixel_matrix = np.transpose(np.array([pixels[i:i + 28] for i in range(0, len(pixels), 28)]))

    
    def draw(self) -> None:
        pg.init()
        surface = pg.display.set_mode((280, 280))
        surface.fill((0, 0, 0))
        pixel_array = pg.PixelArray(surface)
        for x in range(28):
            for y in range(28):
                color = self.pixel_matrix[x][y]
                pixel_array[x*10:x*10 + 10, y*10:y*10 + 10] = (color, color, color)
        pg.display.flip()
        pixel_array.close()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return


def load_dataset(filename: str) -> list[Image]:
    image_list = []
    with open(filename) as csv_file:
        dataset = csv.reader(csv_file, delimiter=',')
        next(dataset)  # skip first elem of dataset, because it is column labels
        image_list = [Image(int(data[0]), [int(data[i]) for i in range(1, len(data))]) for data in dataset]
    return image_list


class NeuralNetwork:
    def __init__(self, sizes: list[int]):
        # sizes[0] is input layer length
        # sizes[-1] is output layer length
        # sizes[1:-1] is length of hidden layers
        self.n = len(sizes)
        self.sizes = sizes
        self.biases = np.random.randn(self.n - 1)  # self.biases[i] == bias for layer i+1
        self.weights = [np.random.randn(sizes[i], sizes[i + 1]) for i in range(self.n - 1)]


def main():
    nn = NeuralNetwork([1, 2, 3])
    print(nn.biases)
    print(nn.weights)

main()