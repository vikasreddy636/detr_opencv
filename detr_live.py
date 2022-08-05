# %%
#opening the image from the url
from PIL import Image
#getting the url
import requests
#displaying the image
import matplotlib.pyplot as plt
#To render high quality images
#%config InlineBackend.figure_format = 'retina'
#for programming the neural network
import torch
from torch import nn
#loading the pretrained neural network
from torchvision.models import resnet50
#for normalising,resizing
import torchvision.transforms as T
#informs all the layers to not to create any computational graph
#because we do not wish to backpropagate for current computations
torch.set_grad_enabled(False);
import cv2
import numpy as np
import random

# %% [markdown]
# minimal implementation of detr

# %%
class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        #Fully Connected Layers form the last few layers in the network
        #The input to the fully connected layer is the output from the final Pooling or Convolutional Layer
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        #convolution layer
        x = self.backbone.conv1(inputs)
        #bn1(bayesian network(notsure))
        x = self.backbone.bn1(x)
        #Activation function
        x = self.backbone.relu(x)
        #pooling operation that calculates the maximum, or largest, value in each patch of each feature map
        x = self.backbone.maxpool(x)

        #Hidden layers
        #ResNet-50 has 3 hidden layers, so we freeze the firstlayer(https://github.com/facebookresearch/detr/issues/494)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        #Concatenates the given sequence of seq tensors in the given dimension
        #All tensors must either have the same shape (except in the concatenating dimension)
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

# %%
#since the length of classes is 91 we use num_classes = 91
detr = DETRdemo(num_classes=91)
#detr = detr.cuda()
#It's useful to be able to load pretrained weights via a single-stop method without making checks if it's a url or a local path.
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)

detr.eval();

# %%
# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
len(CLASSES)

# %%
# standard PyTorch mean-std input image normalization
# [0.485, 0.456, 0.406] means of imagenet data set
# [0.229, 0.224, 0.225] standard deviation of imagenet data set
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#Even if you dont. use the function it works
# for output bounding box post-processing
#for creating a box(not rectangle) 
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    #0.5 is added to find the top and bottom part of the box(not sure)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
#that box with respect to image size
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# %% [markdown]
# x_c, y_c, w, h = 100, 50, 200, 100
# b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#         (x_c + 0.5 * w), (y_c + 0.5 * h)]
# b

# %%
#combining all the above fnctions in detect
def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    #unsqueeze(0):Returns a new tensor with dimension of 1 insterd of original position.
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    # assert takes the input, which when returns true doesn’t do anything and continues the normal flow of execution.
    #but if it is computed to be false, then it raises an AssertionError along with the optional message provided. 
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    #softmax converts vector numbers to vector probabilities.here we gonna use softmax of the last dimension.
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    #why 0.7?
    #there is no specific reason but its mainly used because the algorithm below 0.7 is kind of guss insterd of predicting.
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

# %%

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
#     vid = cv2.imshow('frame',frame)
    img = cv2.resize(img, (800, 800))
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    scores,boxes=detect(im_pil,detr,transform)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    color = list(np.random.random(size=3) * 256)
    for p, (xmin, ymin, xmax, ymax), c in zip(scores, boxes.tolist(), COLORS * 100):
        #cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax-xmin), int(ymax-ymin)), (color), 1)
        #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (color), 2)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        cv2.putText(img, text, (int(xmin), int(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)       
        

    cv2.imshow("test", img)
        
              
    #print(boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindows("test")
                         

# %%


# %%


# %%



