# 代码说明文档

## 1. 介绍
本项目为使用tensorflow1实现的R_TGF算法代码，使用强化学习设计稀疏子图发现算法。对应论文第四章。


### 2. 运行环境

```
cython==0.29.13 
networkx==2.3 
numpy==1.17.3 
pandas==0.25.2 
scipy==1.3.1 
tensorflow-gpu==1.14.0 
gcc==7.4.0（或更高版本）
```


## 3. 运行方法
cd R_TGF
  - 首先编译代码
    ```
    python setup.py build_ext -i
    ```
  - 然后训练模型
    ```
    python train.py
    ```
  - 最后生成稀疏子集
    ```
    python tgf.py
    ```


## 4. 运行步骤
1）使用强化学习模型DQN进行模型训练，得到节点表征
```
CUDA_VISIBLE_DEVICES=gpu_id python train.py
```
在 `FINDER.pyx`文件中修改模型参数
主要调整参数：

2）人工数据集生成稀疏子集
```
CUDA_VISIBLE_DEVICES=-1 python testSynthetic.py 
```


3）真实数据集生成稀疏子集
```
CUDA_VISIBLE_DEVICES=-1 python testReal.py 
```

4）模型保存
将训练好的模型保存在'./models'文件中


5）结果生成
运行tgf.py后得到的结果保存在'./results'文件中

## 5. 数据集
- data: 数据集
  - real 真实数据集
  - synthetic 人工生成数据集

## 6. 文件调用顺序
  1）模型训练阶段
  -train.py————调用Playgame函数，学习强化学习策略————调用Run_Simulator函数，得到action集合

    --Run_Simulator函数————调用PredictwithCurrentQnet函数，学习Q网络—————调用Predict函数，学习节点Q值

  2）模型预测阶段
  -train.py————调用Test函数————调用PredictwithCurrentQnet函数，学习Q网络—————调用Predict函数，得到节点Q值————调用utils.getRobustness函数得到稀疏子集集合

