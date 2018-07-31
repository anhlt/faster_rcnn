import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def imshow(inp, gt_boxes=[], predict_boxes=[], random=False):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig, ax = plt.subplots(1, figsize=(20, 10))

    ax.imshow(inp)
    for i, box in enumerate(gt_boxes):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0],
                                 box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    color = 'b'
    for i, box in enumerate(predict_boxes):
        if random:
            color = np.random.rand(3)
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0],
                                 box[3] - box[1], linewidth=1, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.pause(0.001)  # pause a bit so that plots are updated
