# Face2Vec-with-CelebaDataset
Object oriented implementation with Tensorflow 2


![Face2Vec](https://user-images.githubusercontent.com/35764362/122306214-b8b18480-cf10-11eb-9394-144aa06979d9.gif)


The Face2Vec model is converting face images to vectors which is 40 length. 
The slimcnn model was used in this model, this model provide high accuracy model even you can run on mobile phone. Today's most of known deep learning models are not clearly give a solution about face verification and face analysis. If you have not enough data and computer power you can't imporove a model. And also the human faces are private condition, the users dont want to share their data. The slim cnn module is focusing create a model which is work on mobile phone. 

The training was trained on google colob free version. The implementation of code was prepared by original paper. In this paper, the model architecture is not clearly explained, for instance the model paramethers and layer input size is not match because of the resnet block. Therefore I added (1, 1) conv2d to changing filter size of input without changing the dimension of input.

[https://arxiv.org/abs/1907.02157](https://arxiv.org/abs/1907.02157)
