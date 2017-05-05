# NLPCC 2017 新闻标题分类
### ***预训的embedding 放在[百度云](https://pan.baidu.com/s/1mhPddpu)，可以自行下载。
## 代码运行环境
python2.7 (最好用anaconda2)<br>
tensorflow1.0.0 gpu版本或者cpu版本<br>
建议操作系统:Linux<br>

> Linux 上的环境配置可以参考[Setup Deep Learning enviroment on linux](https://jerrikeph.github.io/setup-deep-learning-enviroment-on-linux.html)。注意要自己在tensorflow网站上找到自己要的版本<br>

	https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl
	https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl

## 快速上手
> 这里只提供在linux 上的上手攻略

在满足上面运行环境之后，可以直接运行.
	
	bash ./quick_run_.sh

就开始训练了。log保存在`./savings/save01/run.log` 里面。
#### 里面做了这些事情：

在命令行中`$tar zxvf nlpcc_data.tar.gz` 解压 nlpcc_data.tar.gz<br>

解压后的目录结构：

	.
	├── char
	│   ├── dev.txt
	│   ├── id2tag.txt
	│   ├── test.txt #测试文件在nlpcc_data中并没有给出，将dev.txt复制一份成test.txt
	│   ├── train.txt
	│   └── vocab.txt
	└── word
	    ├── dev.txt
	    ├── id2tag.txt
	    ├── test.txt #测试文件在nlpcc_data中并没有给出，将dev.txt复制一份成test.txt
	    ├── train.txt
	    ├── vocab.100k
	    └── vocab.all


将`char`或者`word`中的copy到`all_data`中<br>
下面是运行命令，

	#在save01目录中生成一个config文件
	python model.py --weight-path ./savings/save01 
	# 载入./savings/save01中的配置文件并且开始训练
	python model.py --weight-path ./savings/save01 --load-config
	# 载入./savings/save01中的配置文件以及保存在改目录下的训练好的参数进行测试
	python model.py --weight-path ./savings/save01 --load-config --train-test test

`./savings/save01/config`文件可以修改，然后不用执行第一句命令生成配置文件。直接执行第二条命令载入修改好的配置文件。<br>
配置文件长这样：

	[General]
	train_data = ./all_data/train.txt
	val_data = ./all_data/dev.txt
	test_data = ./all_data/test.txt
	vocab_path = ./all_data/vocab.txt
	id2tag_path = ./all_data/id2tag.txt
	embed_path = ./all_data/embed/embedding.
	neural_model = lstm_basic
	pre_trained = False
	vocab_size = 100000
	batch_size = 64
	embed_size = 200
	max_epochs = 50
	early_stopping = 5
	dropout = 0.9
	lr = 0.001
	decay_steps = 500
	decay_rate = 0.9
	class_num = 0
	reg = 0.001
	num_steps = 40
	fnn_numlayers = 1

	[lstm]
	hidden_size = 300
	rnn_numlayers = 1

	[cnn]
	num_filters = 128
	filter_sizes = [3, 4, 5]
	cnn_numlayers = 1
> 配置文件用来配置模型结结构


## 数据描述
char目录中的数据是字符级别的新闻标题<br>
word目录中的数据是词级别的新闻标题 (分词工具为jieba，也可以用其他工具分)<br>
内容类似于：
> finance&nbsp;&nbsp;&nbsp;&nbsp;建 行 按 揭 贷 余 额 超 3 万 亿 还 将 大 力 发 展<br>
society&nbsp;&nbsp;&nbsp;&nbsp;头 号 老 赖 欠 款 2 亿 拆 东 墙 补 西 墙 终 欠 下 2 亿 元<br>
entertainment&nbsp;&nbsp;&nbsp;&nbsp;对 卡 戴 珊 来 说 ， 每 一 次 换 装 都 是 一 次 宣 传<br>
entertainment&nbsp;&nbsp;&nbsp;&nbsp;陈 妍 希 陈 晓 7 月 大 婚 ， 你 看 好 他 俩 吗 ？<br>
car	变 道 和 转 弯 ， 没 让 直 行 车 辆 后 果 可 不 轻<br>
game&nbsp;&nbsp;&nbsp;&nbsp;打 辅 助 位 的 正 统 T D ！ 坦 克 世 界 斯 太 尔 W T 的 战 场 理 解<br>
tech&nbsp;&nbsp;&nbsp;&nbsp;大 数 据 人 才 炙 手 可 热 薪 酬 到 底 有 多 高 ？<br>
travel&nbsp;&nbsp;&nbsp;&nbsp;月 薪 3 0 0 0 元 的 常 州 人 ， 到 这 些 国 家 瞬 间 成 土 豪 ！<br>
history&nbsp;&nbsp;&nbsp;&nbsp;清 朝 灭 亡 时 只 有 2 2 行 省 ， 现 在 却 有 3 4 个 ， 那 些 省 份 是 新 出 的 ？<br>

第一列是label后面是正文，中间用tab隔开。
train.txt, test.txt, dev.txt分别是训练集，测试集以及开发集。id2tag.txt存的标签词典，vocab.txt为辞典。
> 所有的数据都是utf-8格式，以及如果要加入embedding的话，也应该是utf-8格式<br>
> 注意test.txt并没有给出来，可以复制一份dev.txt成test.txt，或者直接改一下config文件

给出的数据是总数据的30%，train跟dev比例为2:1. 其余的70%将会当作测试数据，会在截止日前一周发布。

## 代码描述
	├── Config.py
	├── helper.py
	├── model.py
	
 `Config.py`: 配置处理代码，用于生成或者加载配置文件。<br>
 `helper.py`: 加载数据，加载辞典，计算准确度等等。<br>
### `model.py`: 模型代码。
> 如果需要增加或者修改模型，可以在Model类中的`add_model()`函数中添加一个新的模型模块，或者修改已经有的模型(lstm_basic, cnn_basic, cbow_basic)。
