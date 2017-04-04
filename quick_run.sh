#!/bin/bash

tar zxvf nlpcc_data.tar.gz
cp ./nlpcc_data/char/* ./all_data
cd all_data
cp dev.txt test.txt
cd -
#在save01目录中生成一个config文件
python model.py --weight-path ./savings/save01 
# 载入./savings/save01中的配置文件并且开始训练
python model.py --weight-path ./savings/save01 --load-config
# 载入./savings/save01中的配置文件以及保存在改目录下的训练好的参数进行测试
python model.py --weight-path ./savings/save01 --load-config --train-test test
