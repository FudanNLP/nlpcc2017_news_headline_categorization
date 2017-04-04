import ConfigParser
import json
class Config(object):
    """Holds model hyperparams and data information.
    
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    """General"""
    train_data='./all_data/train.txt'
    val_data='./all_data/dev.txt'
    test_data='./all_data/test.txt'
    vocab_path='./all_data/vocab.txt'
    id2tag_path='./all_data/id2tag.txt'
    embed_path='./all_data/embed/embedding.'

    neural_model = "lstm_basic"
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
    reg = 1e-3
    num_steps = 40
    fnn_numLayers = 1
    
    """lstm"""
    hidden_size = 300
    rnn_numLayers=1
    
    """cnn"""
    num_filters = 128
    filter_sizes = [3, 4, 5]
    cnn_numLayers=1
    
    def saveConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.add_section('General')
        cfg.add_section('lstm')
        cfg.add_section('cnn')
        
        cfg.set('General', 'train_data', self.train_data)
        cfg.set('General', 'val_data', self.val_data)
        cfg.set('General', 'test_data', self.test_data)
        cfg.set('General', 'vocab_path', self.vocab_path)
        cfg.set('General', 'id2tag_path', self.id2tag_path)
        cfg.set('General', 'embed_path', self.embed_path)

        cfg.set('General', 'neural_model', self.neural_model)
        cfg.set('General', 'pre_trained', self.pre_trained)
        cfg.set('General', 'vocab_size', self.vocab_size)
        cfg.set('General', 'batch_size', self.batch_size)
        cfg.set('General', 'embed_size', self.embed_size)
        cfg.set('General', 'max_epochs', self.max_epochs)
        cfg.set('General', 'early_stopping', self.early_stopping)
        cfg.set('General', 'dropout', self.dropout)
        cfg.set('General', 'lr', self.lr)
        cfg.set('General', 'decay_steps', self.decay_steps)
        cfg.set('General', 'decay_rate',self.decay_rate)
        cfg.set('General', 'class_num', self.class_num)
        cfg.set('General', 'reg', self.reg)
        cfg.set('General', 'num_steps', self.num_steps)
        cfg.set('General', 'fnn_numLayers', self.fnn_numLayers)
        
        cfg.set('lstm', 'hidden_size', self.hidden_size)
        cfg.set('lstm', 'rnn_numLayers', self.rnn_numLayers)
        
        cfg.set('cnn', 'num_filters', self.num_filters)
        cfg.set('cnn', 'filter_sizes', self.filter_sizes)
        cfg.set('cnn', 'cnn_numLayers', self.cnn_numLayers)
        
        with open(filePath, 'w') as fd:
            cfg.write(fd)
        
    def loadConfig(self, filePath):
        cfg = ConfigParser.ConfigParser()
        cfg.read(filePath)
        
        self.train_data = cfg.get('General', 'train_data')
        self.val_data = cfg.get('General', 'val_data')
        self.test_data = cfg.get('General', 'test_data')
        self.vocab_path = cfg.get('General', 'vocab_path')
        self.id2tag_path = cfg.get('General', 'id2tag_path')
        self.embed_path = cfg.get('General', 'embed_path')

        self.neural_model = cfg.get('General', 'neural_model')
        self.pre_trained = cfg.getboolean('General', 'pre_trained')
        self.vocab_size = cfg.getint('General', 'vocab_size')
        self.batch_size = cfg.getint('General', 'batch_size')
        self.embed_size = cfg.getint('General', 'embed_size')
        self.max_epochs = cfg.getint('General', 'max_epochs')
        self.early_stopping = cfg.getint('General', 'early_stopping')
        self.dropout = cfg.getfloat('General', 'dropout')
        self.lr = cfg.getfloat('General', 'lr')
        self.decay_steps = cfg.getint('General', 'decay_steps')
        self.decay_rate = cfg.getfloat('General', 'decay_rate')
        self.class_num = cfg.getint('General', 'class_num')
        self.reg = cfg.getfloat('General', 'reg')
        self.num_steps = cfg.getint('General', 'num_steps')
        self.fnn_numLayers = cfg.getint('General', 'fnn_numLayers')
        
        self.hidden_size = cfg.getint('lstm', 'hidden_size')
        self.rnn_numLayers = cfg.getint('lstm', 'rnn_numLayers')
        
        self.num_filters = cfg.getint('cnn', 'num_filters')
        self.filter_sizes = json.loads(cfg.get('cnn', 'filter_sizes'))
        self.num_filters = cfg.getint('cnn', 'cnn_numLayers')
        