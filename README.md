This code is designed for converting old label-cls pair (provided by ImageNet official website) to new label (ID) for torchvision models.

**Solve the issue:

    Use the pretrained models provided by torchvision and label information from ImageNet official website, you will get the wrong prediction so that the performance could be very poor.

If you have got the issue about torchvision's pretrained models like me, this script could be useful.

```python
## Load old/new ID tables

import numpy as np
from imagenet_dict import *
import json

ids = {}   ## Load old ID dict
with open('imagenet_label.txt', 'r') as fp:
    data = fp.readlines()
    for line in data:
        f, id, cls = line.strip('\n').split(' ')
        ids[f]={}
        ids[f]['id'] = int(id) - 1 ## Since it starts with 1
        ids[f]['cls']= cls

idtrans = np.zeros((1000), np.int)

meta={}
with open('folder2ID.json', 'r') as fp:  ## Load new ID table
    data =fp.readline()
    data = json.loads(data)
    for k in list(data.keys()):
        f,cls = data[k]
       
        meta[f]={}
        meta[f]['label']=int(k)
        meta[f]['cls']=cls
        meta[f]['old_id']=ids[f]['id']
        idtrans[meta[f]['old_id']]=int(k)

```
**Generate the train.txt and val.txt (if necessary for your dataloader)

```python
import os

dirs = os.listdir('train')
with open('train.txt', 'w') as fp:
    for d in dirs:
        fns = os.listdir('train/'+d)
        for f in fns:
            if f[-5:]=='.JPEG':
                fp.write('%s %d\n' % (os.path.join(d, f), meta[d]['label']))
                if meta[d]['label']>999:
                    print(d,meta[d])
            
valgt = 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
## ILSVRC2012_val_00050000.JPEG

with open(valgt, 'r') as fp:
    valgt = fp.readlines()

with open('val.txt', 'w') as fp:
    for i in range(50000):
        tid = int(valgt[i].strip('\n'))
        tid = idtrans[tid-1]
        if tid>999:
            print('val', i, tid)
        fp.write('%s %d\n' % ('ILSVRC2012_val_000%05d.JPEG' % (i+1), tid))
                   
```
```asp
Author: Chih-Chung Hsu
Website: https://cchsu.info
e-mail: cchsu@mail.npust.edu.tw
```
