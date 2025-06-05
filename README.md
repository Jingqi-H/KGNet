# KGNet

paper: A Knowledge-Guided Bi-Modal Network for the Classification of Anterior Chamber Angle Images




## Dependencies

This code requires the following:

- python==3.8.3
- numpy==1.18.5
- Pillow==7.2.0
- scikit-learn==0.23.1
- torch==1.7.0
- torchvision==0.8.1



## Training


```
nohup python main.py --max_epoch 300 --p_attri 0.1 --using_seed --save_name KGNet_test >./logs/KGNet_test.log
```


