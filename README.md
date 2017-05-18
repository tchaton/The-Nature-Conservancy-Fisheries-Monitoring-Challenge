# The-Nature-Conservancy-Fisheries-Monitoring-Challenge

Hi there,

I have finished to code a complex deep neural network doing localisation _ refinement _ sotft-attention _ classification . ( It came after several hundreds of work hour, and I began deep learning with this challenge in November ). I have imagined it and I think it might be interesting for you to see it . The localisation part was firstly inspired from a Google paper but I improved it ( when ideas came ).

![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6119/Capture.PNG)
The main ideas are the next ones ( I used KERAS to code it ):
LOCALISATION

# METHOD
1 ) From an image -> Create 2 images of different size and add same augmentation of data process . (3,w,h) - > (3,w1,h1),(3,w2,h2) usually == (3,2*w1,2*h1) but as I modified Resnet architecture, it is not the case .
## Explanation

The inputs shape were [(3,768, 1216),(3,1344,2240)] and I got masks of shape [(1000,18, 32), (1000,36, 64)]. Thanks to UpSampling , I was able to merge them so my networks was able to detect fish from different size. Sliding window were [(224,224),(112,112)]

2 ) Pass it thought a trained model as Resnet50 / VGG16 / VGG19 ( actually I cut it, and modified it in order to have the option "valid" on the AveragePooling2D.

3 ) Map features of this trained network to 5 outputs . Let me explain it here. It is the hard part . The idea is to make the network learn where the fishes are . So we create 5 masks with the value 1 where the fishes are and 0 if not, associated with an equilibrated weight mask ( sum of value of 1 == sum of value of 0 == better learning, so mask of size 100*100 and only I have 1000 1 and 9000 0 , weight of 1 will be 9 and 0 will be 1 ) for the loss function . The 5 masks correspond to one full , one left , one top , one right and one right ( see pictures to understand ) .
## REFINEMENT

4 ) Merge right+left -> rl Merge top + bottom -> tp Merge full + rl + tp -> new full
## SOFT-ATTENTION

5 ) multiply(new_full, Map features) # The features express only where the fishes are , the rest is at almost 0 thanks to soft-attention .
## CLASSIFICATION

6 ) We use the features from soft-attention to make a prediction
NETWORK OUTPUTS

So 2 inputs and 9 outputs = [8 masks (5 : full,left,top,right,bottom) + 3 full ) + prediction class ]
LOSS FUNCTION

I used "mse" to make mask convergence faster. And after , I tried to use [8*mse + categorical_crossentropy ]

## Conclusion :
COMMENTS

In my implementation , the localisation part is really great ( I didn t evaluate it but I evaluate it was able to find the fishes between 70%-90% of the times + able to find several fishes on unseen pictures ). But the network wasn t able to learn the classification part ( I might I have mixed things there ).

I HAVE ADDED SOME PICTURES FOR BETTER UNDERSTANDING:

1 ) ORIGNAL IMAGE

![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6120/Capture1.PNG)

2 ) ONE OF THE INPUT IMAGES AFTER DATA AUGMENTATION I did only flip vertical / horizontal + add random normal noise on part where the fishes weren t -> decrease overfit on boats .

![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6121/Capture3.PNG)

3.a) LEFT : full mask , RIGHT : weigths associated
![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6124/Capture4.PNG)

3.b) LEFT : top mask , RIGHT : weigths associated
![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6122/Capture5.PNG)

3.c) LEFT : right mask , RIGHT : weigths associated
![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6123/Capture6.PNG)

3.d) LEFT : bottom mask , RIGHT : weigths associated
![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6125/Capture7.PNG)

3.e) LEFT : left mask , RIGHT : weigths associated
![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169028/6126/Capture8.PNG)


Here are some results:

![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169040/6127/Capture10.PNG)
![](https://kaggle2.blob.core.windows.net/forum-message-attachments/169040/6128/Capture9.PNG)

# NOTEBOOKS

The fish directory juste need to have a directory with test and train to work (I hope)
The other ones in dump but sometimes interesting researches we made on it.

Best Regards , Thomas Chaton.
