# AlexNet
Pytorch implementation of AlexNet.





## 1. Results



### 1. Cifar10/Cifar100

Note, each image in Cifar10/Cifar100 is an image of size `3*32*32`, which is too small for AlexNet who owns 11*11 convolution kernel to forward pass. So, I modify the size of the first two large filter in the paper to `5*5`. I hope this won't add too much bias.



For the results, 3 parallel experiments are carried out for Cifar10 and Cifar100 respectively:

- Original training setting, learning rate is set to `1e-2` initially, and is decided by 10 when the network stops training (i.e., does not increase for several epochs).
- Training with learning rate of `1e-3` in the whole training.
- Training with learning rate of `1e-4` in the whole training. Note, Cifar100 do not carry this experiments for `1e-3` is too worse.



You can also see the result by running:

```shell
git clone https://github.com/jackwang0108/AlexNet.git
cd AlexNet
tensorboard --logdir ./runs
```



**All the accuracy here is top-1 accuracy**.

#### 1. Setting 1

- Cifar10: 

  - best validation accuracy: **78.7%**

  - best testing accuracy: **78.7%**

  - training accuracy:

    ![training accuracy of cifar10](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322171020190.png)

  - training loss:

    ![training loss of cifar10](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322171105095.png)

- Cifar100:

  - best validation accuracy: **41.3%**

  - best test accuracy: **42.5%**

  - training accuracy:

    ![training accuracy of cifar100](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322171436218.png)

  - training loss:

    ![training loss of cifar100](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322171505204.png)





#### 2. Setting 2

- Cifar10: 

  - best validation accuracy: **75.2%**

  - best testing accuracy: **75.4%**

  - training accuracy:

    ![training accuracy of cifar10](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322171816964.png)

  - training loss:

    ![training loss of cifar10](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322172058077.png)

- Cifar100:

  - best validation accuracy: **35.4%**

  - best test accuracy: **35.7%**

  - training accuracy:

    ![training accuracy of cifar100](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322172151100.png)

  - training loss:

    ![training loss of cifar100](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322172414732.png)





#### 3. Setting 3

- Cifar 10

  - best validation accuracy: **78.6%**

  - best testing accuracy: **79.6%**

  - training accuracy:

    ![training accuracy of cifar10](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322172630487.png)

  - training loss:

    ![training loss of cifar10](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/image-20220322172656352.png)
