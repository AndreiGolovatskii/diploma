import matplotlib.pyplot as plt

def visualize(img, markups = None, figsize=(10, 10)):
    figure = plt.figure(figsize=figsize)
    rows, cols = 2, 2

    if markups is not None:
        figure.add_subplot(rows, cols, 1)

    plt.title('qr-code')
    plt.axis("off")
    plt.imshow(img)

    if markups is not None:
        for i, mark in enumerate(markups.keys()):
            i += 2

            figure.add_subplot(rows, cols, i)
            plt.title(mark)
            plt.axis("off")
            plt.imshow(markups[mark].squeeze(), cmap="gray")
    plt.show()
