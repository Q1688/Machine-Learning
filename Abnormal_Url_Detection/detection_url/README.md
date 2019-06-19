代码作用：
使用RNN神经网络对URL进行处理

配置文件：
config.py相关文件路径的配置

URL向量化的文件：
run_url_main.py是对URL进行向量处理
char2ve.py是根据google的word2vec工具处理URL的方法文件

神经网络文件:
run_model.py运行神经网络
Rnn_Net.py神经网络函数模块
run_Prediction.py是预测数据结果模块

方法函数文件：
utils.py存放额外的方法函数模块

model_file文件存放
    checkpoint文件会记录保存信息，通过它可以定位最新保存的模型
    .meta文件保存了当前图结构
    .data文件保存了当前参数名和值
    .index文件保存了辅助索引信息
	
代码不足之处：
	1.无法进行实时数据预测
	2.不是分布式处理
