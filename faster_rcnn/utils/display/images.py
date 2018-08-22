import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ...config import cfg


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


def result_show(inp, predicted_boxes=[], classes=[], scores=[]):
    font = ImageFont.truetype(
        font=cfg.FONT_PATH, size=np.floor(3e-2 * 600 + 0.5).astype('int32'))

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    inp *= 255.0 / inp.max()

    fig, ax = plt.subplots(1, figsize=(20, 10))

    image = Image.fromarray(inp.astype('uint8'))

    draw = ImageDraw.Draw(image)
    for predicted_box, predicted_class, score in zip(predicted_boxes, classes, scores):
        label = '{} {:.2f}'.format(predicted_class, score)

        label_size = draw.textsize(label, font)

        left, top, right, bottom = predicted_box
        print(predicted_class, (left, top), (right, bottom))

        for i in range(3):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=(255, 255, 0))
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)])
        draw.text(text_origin, label, fill=(100, 100, 255), font=font)
    del draw
    ax.imshow(image)
