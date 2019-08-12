算法使用：

    构建cnn卷积神经网络进行分别文章类型


文件介绍：

原始数据文件：

   train.txt（训练）,   val.txt（验证）,    test.txt（预测）
 

清洗之后的数据文件：

   训练数据标签文件：clean_label.txt
   训练数据内容文件：clean_content.txt
   验证数据标签文件：val_clean_label.txt
   验证数据内容文件：val_clean_content.txt
   预测数据标签文件：test_clean_label.txt
   预测数据内容文件：test_clean_content.txt
   
   
配置文件：

   config.py相关文件参数路径配置
   

stopword文件夹：

   myjieba.txt文件是自己制作的结巴分词库
   stopword.txt文件是停用词库


main文件夹步骤：

   1.运行train_word2vec.py进行构建词汇表等，每次添加新的训练数据集时需要重新构建运行
   2.运行train.py进行数据训练/验证/预测处理
   3.运行prediction.py对数据进行预测，预测数据来源方式需要自己更写


缺点：

   自己制作的停用词库较多和结巴分词耗时
   本人对安全知识不是太熟悉，数据标签请别人帮忙弄得，但标签比较混乱，不建议使用此处数据
   把分类出的文本和word2vc文件夹中的文件删除了，那时对数据的清洗产生的，不同的数据文件内容不同，此处避免干扰


注意点：

   对数据进行清洗是重点，此处放置一个链接，讲解挺好的：http://www.sohu.com/a/259646646_717210
   
