# VGG-16 Architecture

## Contents

* [paper](Paper.pdf)
* [implementation](model.py)

## Summary
![1_3-TqqkRQ4rWLOMX-gvkYwA](https://user-images.githubusercontent.com/89085916/169806039-fc80bb1c-e77d-404a-b1ea-4e42582a6bba.png)

### Network Architecture

1. Convolution using 64 filters  
2. Convolution using 64 filters + Max pooling  
3. Convolution using 128 filters  
4. Convolution using 128 filters + Max pooling  
5. Convolution using 256 filters  
6. Convolution using 256 filters  
7. Convolution using 256 filters + Max pooling  
8. Convolution using 512 filters  
9. Convolution using 512 filters  
10. Convolution using 512 filters+Max pooling  
11. Convolution using 512 filters  
12. Convolution using 512 filters  
13. Convolution using 512 filters+Max pooling  
14. Fully connected with 4096 nodes  
15. Fully connected with 4096 nodes  
16. Output layer with Softmax activation with 10 nodes  
[[source]](Paper.pdf)

## CIFAR-10 Dataset
A very popular dataset which contains 10 different objects.   
  
![images](https://user-images.githubusercontent.com/89085916/169703589-40e18144-5f3a-4f5a-ae5f-70f43c66dfc7.jpg)
