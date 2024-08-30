import numpy as np
import imageio
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image = imageio.imread('./data/rubber_whale.png')
    print("shape of image:", image.shape)
    fig = plt.figure()
    plt.imshow(image)

    green_screen_mask = (image[:, :, 1] > 100) & (image[:, :, 2] > 50)
    print("shape of green_screen_mask:", green_screen_mask.shape)

    image[green_screen_mask] = [0, 0, 0] # Remove green background
    fig2 = plt.figure()
    plt.imshow(image)
    plt.show()

    # pause, waiting for key press
    key = input("Press any key to exit")