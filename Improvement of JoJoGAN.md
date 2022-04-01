# Improvement of JoJoGAN

## Generator

- conv层的noise

​	原版：取与input无关的无参数正态分布

​	改进：从input feature计算出特定参数，考虑为均值$\mu$和方差$\sigma^2$,然后采样得到正态噪声：

1. 静态

   通过矩阵理论计算input相关属性，当做正态分布的均值和方差

2. 动态

​		直接丢给神经网络训练得到均值和方差

- 网络结构

  网络结构原版使用styleConv作残差堆叠，可采用加权残差

## Style Mixture

- 原版同样使用无参数正态噪声
- mixture policy可替换



## Discriminator & Loss

- 修改Discriminator的网络结构

  添加AdaIN归一化层和encoder decoder模块

- 替换L1 Loss或尝试使用其他evaluation metric

  

## Imagine Generation

原版使用e4e_projection作GAN Inversion，然后丢给Generatorzuo作图片生成，在此不作其他trick。