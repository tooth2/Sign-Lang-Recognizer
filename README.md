# Sign-Lang-Recognizer
American Sign Language Recognizer

## 1. Dataset
![Sign Lang](images/signs_data_kiank.png)

## 2. Build Model
### Building a Residual Network
In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:
![skip_connection](images/skip_connection_kiank.png)
![Resnet](images/resnet_kiank.png)

### 2.1 - The identity block
The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say $a^{[l]}$) has the same dimension as the output activation (say $a^{[l+2]}$). To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:
 - Identity block. Skip connection "skips over" 2 layers.
 ![identity block](images/idblock2_kiank.png)
 
 The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step.the skip connection "skips over" 3 hidden layers rather than 2 layers. It looks like this: 
 - Identity block. Skip connection "skips over" 3 layers.
 ![identity block2](images/idblock3_kiank.png)
 
 ### 2.2 - The convolutional block
 the ResNet "convolutional block" is the other type of block. You can use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path:
 ![conv block](images/convblock_kiank.png)
 
 ### 3 - Building your first ResNet model (50 layers)
ResNet50 is a powerful model for image classification when it is trained for an adequate number of iterations. The following figure describes in detail the architecture of this neural network. "ID BLOCK" in the diagram stands for "Identity block," and "ID BLOCK x3" means you should stack 3 identity blocks together.
![resnet50](images/resnet_kiank.png)
```
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Take aways
- Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.
- The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function.
- There are two main type of blocks: The identity block and the convolutional block.
- Very deep Residual Networks are built by stacking these blocks together

### Reference
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [Francois Chollet's github ](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py)
