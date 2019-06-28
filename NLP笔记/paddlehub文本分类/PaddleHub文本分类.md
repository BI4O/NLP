# PaddleHub文本分类

- ### 执行Fine-tune有两种方式

  1. 命令行执行.sh脚本
  2. python代码

- ### 命令行fine-tune

  首先要确保git clone了paddlepaddle和paddlehub然后cd到PaddleHub/demo/**text-classification**/

  ~~~shell
  sh run_classifier.sh \
      # 模型相关
      --batch_size: 批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数
      --learning_rate: Finetune的最大学习率
      --weight_decay: 控制正则项力度的参数，用于防止过拟合，默认为0.01
      --warmup_proportion: 学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0
      --num_epoch: Finetune迭代的轮数
      --max_seq_len: ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数
  
      # 任务相关
      --checkpoint_dir: 模型保存路径，PaddleHub会自动保存验证集上表现最好的模型
      --dataset: 有三个参数可选，分别代表3个不同的分类任务; 分别是 chnsenticorp, lcqmc, nlpcc_dbqa
  ~~~

  

- ### python代码方式fine-tune（详细介绍）

  1. #### 第一步：加载预训练模型

     ~~~python
     module = hub.Module(name="ernie")
     # 拆包得到：输入形式，输出形式，预训练模型ernie
     inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
     ~~~

     其中这个max_seq_len参数是可以调整的，建议128，name参数决定的是使用哪个模型，如果不用`ernie`下面的模型都可以选择

     | 模型名                        | PaddleHub Module                                      |
     | ----------------------------- | ----------------------------------------------------- |
     | ERNIE, Chinese                | `hub.Module(name='ernie')`                            |
     | BERT-Base, Uncased            | `hub.Module(name='bert_uncased_L-12_H-768_A-12')`     |
     | BERT-Large, Uncased           | `hub.Module(name='bert_uncased_L-24_H-1024_A-16')`    |
     | BERT-Base, Cased              | `hub.Module(name='bert_cased_L-12_H-768_A-12')`       |
     | BERT-Large, Cased             | `hub.Module(name='bert_cased_L-24_H-1024_A-16')`      |
     | BERT-Base, Multilingual Cased | `hub.Module(nane='bert_multi_cased_L-12_H-768_A-12')` |
     | BERT-Base, Chinese            | `hub.Module(name='bert_chinese_L-12_H-768_A-12')`     |

     例如使用bert中文模型

     ~~~python
     module = hub.Module(name="bert_chinese_L-12_H-768_A-12")
     ~~~

     

  2. #### 第二步：准备数据集并使用ClassifyReader来读取数据

     ~~~python
     # 实例化你的数据集类，生成一个数据集对象dataset
     # 这里是一个数据的类，后面需要的话可以进行改写
     dataset = hub.dataset.ChnSentiCorp()
     # 读取这个数据集用于fine-tune
     reader = hub.reader.ClassifyReader(
         # 而数据来源就是指向这个类生成的对象
         dataset=dataset,
         # 指定词表
         vocab_path=module.get_vocab_path(),
         # 指定最大序列长度，多出的将会被padding
       max_seq_len=128)
     ~~~

     - #####  这个reader做事情：

       ClassifyReader中的`data_generator`会自动按照模型对应词表（`module.get_vocab_path`）进行切词，这个你指定的词表和最大序列长度就决定了你的数据的切词方式，顺带提一下它会以迭代器的方式返回模型需要的所有Tensor格式，包括`input_ids`，`position_ids`，`segment_id`与序列对应的mask `input_mask`，如果这些你都不懂，那也没有关系，因为你只要知道输入的.tsv里面的数据格式是什么就够了，下面会讲

     - #####  分析样本数据处理的py文件源码：

       指定要用什么数据集的时候，如果你没有这个数据集，那么执行的时候会自动帮你下载到用户目录`user/paddlehub/dataset`下，这点在chnsenticorp.py中有说明（这个py文件在PaddleHub/paddlehub/dataset/chnsenticorp.py)
  
       ~~~python
       _DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz"
       
       # DATA_HOME就是上面提到的user/paddlehub/dataset
       class ChnSentiCorp(HubDataset):
           def __init__(self):
               # 首先指定了数据的目录是user/paddlehub/dataset/chnsenticorp
               self.dataset_dir = os.path.join(DATA_HOME, "chnsenticorp")
               # 如果目录中找不到这个数据集，就下载一个
               if not os.path.exists(self.dataset_dir):
                   	ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                       url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
               # 如果找到了这个数据集
               else:
                 logger.info("Dataset {} already cached.".format(self.dataset_dir))
       ~~~

       ##### 下载后的chnsenticrop数据：

       ![1561600421147](C:\Users\CJB\AppData\Roaming\Typora\typora-user-images\1561600421147.png)

       

       ##### 然后用记事本打开较小的test.tsv看看数据的格式：

       ![1561600796026](C:\Users\CJB\AppData\Roaming\Typora\typora-user-images\1561600796026.png)

       

       ##### 找一找规律：

       除了第一个是`label` `\t` `text_a` 以外，后面的都是`0或1` `\t` `文本评价` 如果把`1\t评价` 作为一条记录的话，那么除了第一条是格式说明以外，剩下的都是记录，注意记录与记录之间是没有任何字符的（空格也没有），而1代表的是积极评价，0代表的是消极评价

       

     - ##### 回到代码，研究一下chnsenticorp.py是怎么读取数据的
  
       ~~~python
       # 导包代码略去
       DATA_HOME = 'C://Users/CJB/.paddlehub/dataset'
       
       class ChnSentiCorp(HubDataset):
           def __init__(self):
               # 下载数据代码段略去
               self.dataset_dir = os.path.join(DATA_HOME, "chnsenticorp")
               # 这里初始化的时候加载三个文件的方法
               self._load_train_examples()
               self._load_test_examples()
               self._load_dev_examples()
               
           # 以加载test为例看一下，dev&train同理
           def _load_test_examples(self):
               # 定位到了这个文件路径
               self.test_file = os.path.join(self.dataset_dir, "test.tsv")
               # 读这个路径下的tsv文件
               self.test_examples = self._read_tsv(self.test_file)
               
           # 看看是怎么读tsv文件的
           def _read_tsv(self, input_file, quotechar=None):
               """Reads a tab separated value file."""
               with codecs.open(input_file, "r", encoding="UTF-8") as f:
                   # 原来是以\t来分割每一条记录
                   reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                   examples = []
                   seq_id = 0
                   # label \t text_a,这个不能当记录，先放出来
                   header = next(reader)  # skip header
                   # 然后再遍历reader把后面的记录取出来
                   # 这里可以看到，每个example记录由seq_id+line[0]+line[1]组成
                   # 分别对应着这条example记录的guid, label, text_a
                   for line in reader:
                       example = InputExample(
                           guid=seq_id, label=line[0], text_a=line[1])
                       seq_id += 1
                       # 把每条example记录一次放进examples这个list中
                       examples.append(example)
                     
                   # 返回这个带有每条记录的list
               return examples
       ~~~

       好了，这下我们终于知道，原来这个类做的，就是把tsv文件读取你便成了examples这样list列表，一共由三个，分别来自train，test，dev
  
       然后这个py文件还会在文件处理完后打印出处理好的文本
       
       ~~~python
       if __name__ == "__main__":
         ds = ChnSentiCorp()
           for e in ds.get_train_examples():
           print("{}\t{}\t{}\t{}".format(e.guid, e.text_a, e.text_b, e.label))
       ~~~

       至此，我们已经弄懂了数据是怎么被读进来的了，而本次文本分类中，ernie模型要吃的就是这三个包含所有样本的examples list
     
       

  3. ### 第三步：构建网络模型并创建分类迁移任务
  
     记得第一步中拆包得到的inputs，outputs，program吗，这里要用
  
     ~~~python
     # NOTE: 必须使用fluid.program_guard接口传入Module返回的预训练模型program
     
     # 模型是有很多的output类型的，这就像神经网络模型的最后一层
     # 因为我们本次的分类是二分类任务，所以最后只要一个unit点就好了
     pooled_output = outputs["pooled_output"]
     
     # 这是预训练模型根据这个任务要接的线性分类器
     # 这个线性分类器有一个output unit,而output的label是[0,1]
     cls_task = hub.create_text_cls_task(
         feature=pooled_output, 
         num_classes=dataset.num_labels
     )
     
     # 模型的输入就类似神经网络模型的输入层
     # 而第二步的reader把数据进行处理之后每个样本有5个features,所以输层有5个units
     # feed_list的Tensor顺序不可以调整
     feed_list = [
         inputs["input_ids"].name, 
         inputs["position_ids"].name,
       inputs["segment_ids"].name, 
         inputs["input_mask"].name, 
       cls_task.variable("label").name
     ]
     ~~~
     
      - Note的意思大概就是，这个program一定要拿第一步载入模型时候生成的那个变量，而不是从其他框架拿过来的预训练模型
      - `outputs["pooled_output"]`返回了与训练模型中对应的[CLS]向量，代表了分类信息
      - `feed_list`中的inputs参数指明了与训练模型的输入tensor的顺序，与ClassifyReader返回的结果一致
     - `create_text_cls_task`也就是相当于构建了ernie要接的分类器的结构，这个分类器在fine-tune阶段将会与与ernie模型一起反向传播训练更新参数，不同的是，这个分类器的参数是从头训练的，而ernie是有已经与训练过的参数的
     - 这个`dataset.num_lables`是一个常数，也即是类别数，在第二步中用到的chnsenticorp.py文件中有详细的说明
     
     ~~~python
     class ChnSentiCorp(HubDataset):
         ...
         def get_labels(self):
             return ["0", "1"]
     
         @property
         def num_labels(self):
             """
                Return the number of labels in the dataset.
                  """
             return len(self.get_labels())
         ...
     ~~~
     
     
  
  4. ### 第四步：选择优化策略并开始Fine-tune
  
     ~~~python
     # 指定optimizer的优化策略
     strategy = hub.AdamWeightDecayStrategy(
         # 最大学习率
         learning_rate=5e-5,
         # 正则化系数，如果有过拟合风险可以调高这个参数
         weight_decay=0.01,
         # 学习率变化率，例如设为0.1那么会在每次epoch的前10%中线性增长到最大学习率
         warmup_proportion=0.0,
         # =linear_decay：学习率在最高点后以线性方式衰减
         # =noam_decay：学习率在最高点以后以多项式形式衰减
         lr_scheduler="linear_decay",
     )
     
     # 配置fine-tune信息
     config = hub.RunConfig(
         # fine-tune是否使用cuda加速，没有cuda选False
         use_cuda=True, 
         num_epoch=3, 
         batch_size=32, 
         strategy=strategy
     )
     
     # 启动fine-tune
     hub.finetune_and_eval(
         # cls_task包含了分类器的所有信息
         task=cls_task, 
         # reader包含了经过切词的样本，每个样本含有5个特征的
         data_reader=reader, 
         # feed_list包含了ernie+分类器整个模型的输入，是5个units
       feed_list=feed_list, 
         # config包含了训练的优化器及其配置，遍历样本次数，batch大小
       config=config
     )
     ~~~
     
      ###### 运行配置
     
      `RunConfig` 主要控制Finetune的训练，包含以下可控制的参数:
     
      - `log_interval`: 进度日志打印间隔，默认每10个step打印一次
     - `eval_interval`: 模型评估的间隔，默认每100个step评估一次验证集
      - `save_ckpt_interval`: 模型保存间隔，请根据任务大小配置，默认只保存验证集效果最好的模型和训练结束的模型
     - `use_cuda`: 是否使用GPU训练，默认为False
      - `checkpoint_dir`: 模型checkpoint保存路径, 若用户没有指定，程序会自动生成
     - `num_epoch`: finetune的轮数
      - `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size
     - `enable_memory_optim`: 是否使用内存优化， 默认为True
      - `strategy`: Finetune优化策略
     
      当你看到这样的输出的时候，就代表模型已经训练完成啦
     
     ~~~shell
     运行耗时: 9分18秒640毫秒
      ...
      ...
     2019-06-27 17:30:56,606-INFO: PaddleHub finetune finished.
     ~~~
     
  5. ### 第五步：训练过程可视化
  
       其中${HOST_IP}为本机IP地址，如本机IP地址为192.168.0.1，用浏览器打开192.168.0.1:8040，其中8040为端口号，即可看到训练过程中指标的变化情况

  

  6. ### 第六步：模型预测
  
     ###### 直接在notebook中运行以下代码，
  
     ~~~python
     !python predict.py --checkpoint_dir "/home/aistudio/ckpt_20190627172138/best_model" --max_seq_len 128
     ~~~
  
       -  其中这个`checkpoint_dir`**最好使用绝对路径来指向！！用绝对路径！！用绝对路径！！用绝**
         因为启动`predict.py`必须在`predict.py`的路径下，但是你的`best_model`却**不一定在这个文件夹中**，它是由执行`hub.finetune_and_eval()`方法的时候决定的，你可以在这个方法中指定`checkpoint_dir=/path/to/best_model`，可是我没有指定了额，所以后来**生成到根目录去了**
         
     - max_seq_len是ERNIE模型的最大序列长度，*请与训练时配置的参数保持一致*
  
       当看到有这样的输出的时候，就代表已经用上训练好的模型了
  
       ~~~shell
       运行耗时: 17分8秒519毫秒
       ...
       text=房间不错,只是上网速度慢得无法忍受,打开一个网页要等半小时,连邮件都无法收。另前台工作人员服务态度是很好，只是效率有得改善。	label=1	predict=1
       text=挺失望的,还不如买一本张爱玲文集呢,以<色戒>命名,可这篇文章仅仅10多页,且无头无尾的,完全比不上里面的任意一篇其它文章.	label=0	predict=0
       accuracy = 0.943333
       ~~~
  
       预测准确率：94.3% 
  
       
  
  7. ### 第七步
  
     这样就完了吗，当然不是，我可是要用模型来打比赛的耶( •̀ ω •́ )y，我们来看一下怎么把预测好的结果取出来，这就需要研究一下predict.py这个文件的代码了
  
     