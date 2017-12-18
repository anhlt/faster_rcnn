## Faster RCNN

An another pytorch implementation of Faster RCNN base on [https://github.com/longcw/faster_rcnn_pytorch](longcw/faster_rcnn_pytorch), with rewriten data pre-process module and a lot of helpful debug messages.


## Installation

1. Install docker and Nvidia-docker
2. Git clone
    git clone https:github.com/anhlt/faster_rcnn
    
3. Using `docker-compose`
    docker-compose up --build
    
    
## Modules

### Dataset module

Extend the `torchvision.datasets.CocoDetection` module, and add extracting bounding box function. Image data will be automatically resize, and normalize

```python
import os
import torchvision.transforms as transforms
from faster_rcnn.utils.dataset import CocoData
from faster_rcnn.utils.data_generator import CocoGenerator
from faster_rcnn.utils.data_generator import Enqueuer

dataDir = './data/mscoco'
dataType = 'train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

images_dir = os.path.join(dataDir,'images', dataType)
cap = CocoData(root = images_dir,
                        annFile = annFile,
              )
```
You can verify your dataset module with this method
```python               
def imshow(inp, gt_boxes=[], predict_boxes = []):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    fig,ax = plt.subplots(1, figsize=(20, 10))

    ax.imshow(inp)
    for i, box in enumerate(gt_boxes):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1]  ,linewidth=2,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        
    for i, box in enumerate(predict_boxes):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1]  ,linewidth=1,edgecolor='g',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

data = cap[13499]
im = data['tensor']
gt_boxes =  data['boxes']
imshow(im[0], gt_boxes)
```

![image](https://raw.githubusercontent.com/anhlt/faster_rcnn/master/docs/images/image1.png)
