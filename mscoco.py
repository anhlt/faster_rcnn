# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('Agg')


# In[2]:


from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from faster_rcnn.utils.cython_bbox import bbox_overlaps
from pycrayon import CrayonClient

import cPickle
from torch.optim import SGD, RMSprop, Adam
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from faster_rcnn.utils.datasets.adapter import convert_data
from faster_rcnn.utils.evaluate.metter import AverageMeter
from faster_rcnn.utils.display.images import imshow, result_show


# In[3]:


with open('sorted_index', 'rb') as fp:
    sorted_index = cPickle.load(fp)


# ### Đọc dữ liệu từ MS COCO dataset
# 

# In[4]:


import os
import torchvision.transforms as transforms
from faster_rcnn.utils.datasets.mscoco.dataset import CocoData
from faster_rcnn.utils.datasets.data_generator import CocoGenerator
from faster_rcnn.utils.datasets.data_generator import Enqueuer

dataDir = './data/mscoco'
dataType = 'train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
batch_size = 10

images_dir = os.path.join(dataDir,'images', dataType)
cap = CocoData(root = images_dir, annFile = annFile)

data_gen = CocoGenerator(data=cap, sorted_index=sorted_index, batch_size=batch_size)
queue = Enqueuer(generator=data_gen)
queue.start(max_queue_size=10, workers=2)
train_data_generator = queue.get()


# In[5]:


dataDir = './data/mscoco'
dataType = 'val2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
batch_size = 8

images_dir = os.path.join(dataDir,'images', dataType)
val_cap = CocoData(root = images_dir, annFile = annFile)

val_data_gen = CocoGenerator(data=val_cap, batch_size=batch_size, shuffle=True, seed=2)
val_queue = Enqueuer(generator=val_data_gen)
val_queue.start(max_queue_size=10, workers=2)
val_data_generator = val_queue.get()


# Thử hiển thị ảnh cùng các bounding boxes

# In[6]:


from faster_rcnn.faster_rcnn import FastRCNN


# ### Tính toán feed-forward
# 
# 
# Chúng ta sử dụng một ảnh có kích thước đầu vào là  `(width , height) = (600, 800)`
# 
# Input:
#     - im_data : 
#         kích thước : (batch_size, dim, witdh, height)
#     - ground_boxes: 
#         kích thước (n, 4)
#         

# In[7]:


categories = ['__background__'] + [x['name'] for x in cap.coco.loadCats(cap.coco.getCatIds())]


# In[8]:


net = FastRCNN(categories, debug=False)
net.cuda()


# In[9]:


param = filter(lambda x: x.requires_grad, net.parameters())
optimizer = SGD(param, lr=1e-3, momentum=0.9, weight_decay=0.0005)
exp_lr_scheduler = StepLR(optimizer, step_size=1000, gamma=0.95)


# In[10]:


def evaluate(data_gen ,model,tensorboard_client, steps_per_epoch=2000, current_epoch=0, current_step=0):
    
    model.eval()
    val_loss = AverageMeter()
    val_cross_entropy = AverageMeter()
    val_loss_box = AverageMeter()
    val_rpn_loss = AverageMeter()
    with torch.no_grad(): 
        for step in range(1, steps_per_epoch +1):
            blobs = data_gen.next()
            batch_tensor, im_info, batch_boxes, batch_boxes_index = convert_data(blobs)
            model(batch_tensor, im_info, batch_boxes, batch_boxes_index)
            loss = model.loss
            val_loss_box.update(model.loss_box.item())
            val_cross_entropy.update(model.cross_entropy.item())
            val_loss.update(loss.item())
            val_rpn_loss.update(model.rpn.loss.item())
            del loss
            del batch_tensor
            del blobs
            
            if step % 30 == 1:
                log_text = 'epoch: %d : step %d,  val_loss: %.4f at %s' % (
                    current_epoch + 1, step , train_loss.avg, datetime.now().strftime('%m\%d_%H:%M'))
                print(log_text)

    log_text = 'val_loss: %.4f at epoch %d' % (val_loss.avg, current_epoch + 1)
    print(log_text)
    tensorboard_client.add_scalar_value('val_loss', val_loss.avg, step=current_step)
    tensorboard_client.add_scalar_value('val_rpn_loss', val_rpn_loss.avg, step=current_step)
    tensorboard_client.add_scalar_value('val_cross_entropy', val_cross_entropy.avg, step=current_step)
    tensorboard_client.add_scalar_value('val_loss_box', val_loss_box.avg, step=current_step)


# In[11]:


def train(data_gen ,model, tensorboard_client, metters, optimizer, lr_scheduler,steps_per_epoch=2000, current_epoch=0, current_step=0):
    model.train()
    train_loss , cross_entropy , loss_box, rpn_loss = metters
    for step in range(1, steps_per_epoch +1):
        lr_scheduler.step()        
        blobs = data_gen.next()
        batch_tensor, im_info, batch_boxes, batch_boxes_index = convert_data(blobs)
        optimizer.zero_grad()
        model(batch_tensor, im_info, batch_boxes, batch_boxes_index)
        cross_entropy.update(model.cross_entropy.item())
        loss_box.update(model.loss_box.item())
        train_loss.update(model.loss.item())
        rpn_loss.update(model.rpn.loss.item())
        loss = model.loss
        loss.backward()
        optimizer.step()

        current_step = current_epoch * steps_per_epoch + step
        if step % 30 == 1:
            log_text = 'epoch: %d : step %d,  loss: %.4f at %s' % (
                current_epoch + 1, step , train_loss.avg, datetime.now().strftime('%m/%d_%H:%M'))
            print(log_text)

        if step % 100 == 0:
            tensorboard_client.add_scalar_value('train_loss', train_loss.avg, step=current_step)
            tensorboard_client.add_scalar_value('rpn_loss', rpn_loss.avg, step=current_step)
            tensorboard_client.add_scalar_value('cross_entropy', cross_entropy.avg, step=current_step)
            tensorboard_client.add_scalar_value('loss_box', loss_box.avg, step=current_step)
    


# In[12]:


def training(train_data_gen, val_data_gen, optimizer, lr_scheduler ,model, epochs=10):
    
    exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%Ms')
    cc = CrayonClient(hostname="crayon", port=8889)
    exp = cc.create_experiment(exp_name)
    
    
    train_loss = AverageMeter()
    cross_entropy = AverageMeter()
    loss_box = AverageMeter()
    rpn_loss = AverageMeter()
    metters = (train_loss , cross_entropy , loss_box, rpn_loss)
    steps_config = {
        'train_step_per_epoch' : 8000,
        'val_step_per_epoch' : 5000
    }
    
    for epoch in range(epochs):
        train(data_gen ,model, exp, metters, optimizer, lr_scheduler, steps_per_epoch=steps_config['train_step_per_epoch'], current_epoch=epoch)
        current_step = steps_config['train_step_per_epoch'] * (epoch + 1)
        #torch.save(model.state_dict(), './checkpoints/faster_model_at_epoch_%d.pkl' % (epoch + 1)) 

#         evaluate(data_gen ,model,exp, steps_per_epoch=steps_config['val_step_per_epoch'], current_epoch=epoch, current_step=current_step)


# In[ ]:
training(train_data_generator, val_data_generator ,optimizer=optimizer,lr_scheduler=exp_lr_scheduler, model=net, epochs=10)


# In[ ]:




# In[ ]:
