# NLP面经准备

- #### 选好方向投，NLP分为偏科研research和偏业务product的

  - 偏科研的需要对算法的原理有比较好的理解，需要现在或将来的产品孵化算法接口
  - 偏业务的需要代码能力好，可以看到自己的成果快速反应在产品的用户体验和经济效益上
  - 如何判断一个部门偏业务还是偏研究呢
    - 偏研究：xx研究院，AI lab，实验室
    - 偏业务/产品：xx事业部，智能客服部，服务部，有道事业部，达摩院AI lab

- #### 笔试与面试中的笔试准备

  - ##### 首先强调基本的编程题，这是要锻炼出来的：

    面试中的白板编程题一般都超级简单，基本是leetcode上简单界别的原题，偶尔会有中等难度的题目，leetcode中刷50-100题即可，**编程语言实现用python就可以了**

  - ##### 计算机专业题目：

    计算机网络，操作系统，计算机组成原理，数据库，编程语言，设计模式之类的基础题一般很少出，出也是选择题

  - ##### 矩阵论，概率统计这些数学课：

    校招的时候比计算机基础还要考得多的题目，如果精力有限那么就只复习最优化问题吧（梯度下降，牛顿法）

- ### 简历的准备

  - #### 简历占据面试中的篇幅最大

    不仅是面试的敲门砖，基本贯穿了整个面试过程，不加白板编程的时间，**简历内容会占据面试8-10成的时间**

  - #### 简历必须整体内容于岗位需求一致

    这是最容易被忽略却又最关键的一条，做简历跟写文章很像，要围绕一根主线展开，如果发现电路也做，视觉也做，推荐也做，NLP也做，很容易被打上“跟岗位不match”或者“这孩子做事浮躁”的tag

    如果已经有一连串paper的简历，那就不做建议了，但是如果比赛，文章少的话，完全可以把简历做的有的放矢，**match的内容展开写，不match的内容一笔带过或者直接不写**，这样面试官也省的一条条甄别信息

  - #### 写项目的技巧

    可以尝试把最有信心在面试中谈起的经历所在的板块写在最前面（仅次于教育经历）并用配色突出这一条，这样可以聚焦非常多的面试火力，甚至几轮的面试官都会同一问你这条经历，当然这样的副作用是，如果对面试管来说这条经历不亮，那面试基本挂掉一半了

  - #### 写熟悉的方向

    nlp也是有非常多的方向的，建议单独列一个板块列一下自己研究过的算法问题，比如分为两级，第一级讲方向（比如对话系统）第二级讲具体研究的子问题（比如聊天的一致性问题）这样可以避免面试官对你进行天马行空式的考察，毕竟时间有限，难以研究的面面俱到，该板块会贡献大量的关键词，这些关键词决定了面试官对你的考察范围

  - ### 论文

    顶会的非常难，但是水会的尽量拿一些

  - ### 比赛

    比赛比论文的性价比更高，但是小企业小机构般的队伍参赛的小比赛就不要写了，除了NLP各大顶会和Kaggle的比赛，还可以关注哥哥互联网大厂举办的NLP比赛，比如微软的编程之美挑战赛（去年是问答bot），百度的机器阅读理解大赛，阿里的天池系列比赛，亲测在大厂中很有效（非举办方的互联网公司一般也会关注友商的比赛的）

    **一定要打跟目标岗位match的比赛**，底线是NLP比赛，比如你想做chatbox，却打一些数据挖掘类的比赛，那哪怕top5也意义不大，但是这时的文本匹配，生成，问答相关的比赛哪怕是排名一般（队伍数前10%）也完全可以写上去。已经有好的名次但是match匹配程度不高的怎么办？一句带过吧，否则会让人觉得你跟岗位不match

    关于打比赛，如果是做NLP，千万不要堆开源模型做ensemble上分，这样虽然会为你争取到面试机会，但是基本没有任何创新，只会让面试官觉得你是个优秀的板砖工程师或者调参小能手，**对于面试来说，优秀的单模型超级好用**，另外最好把定会SOTA也拿到比赛数据集上跑一下，这样面试更有说服力。

  - ### 实习

    好的大厂实习非常加分

  - ### 其他

    其实论文/比赛/大厂实习这三个都不是必须的，但是最好有其中之一，从身边的例子来看，只要有其一一般bat核心部门或者核心业务部门至少能拿一个offer

    如果很不幸成为了“三无”人员，那么一定要保证扎实的数学，nlp，coding能力和至少一个研究方向的专精，能够在面试时候表现出超出简历描述的能力，这样也是很打动面试官的，毕竟都想找一个潜力股

### 其他干货

- #### 面试中的基础算法知识

  小编曾经挤出时间很努力的手撸了一遍LR，最大熵，决策树，朴素贝叶斯，svm，em，hmm，crr，结果从来没有被问到过，是从来

  然而很奇葩的被问倒了tcp的三次握手，hadoop的shuffle机制，linux的find命令怎么用（from 今日头条）

  虽然实现NLP的方法基本离不开机器学习的神经网络，但是如果按照前面的讲的准备简历内容，其实在NLP岗中很少直接考察ML和NN的理论知识，那考察什么呢？当然是关键词，**所以总结一下自己简历的关键词，然后展开复习吧**

  - #### 小编简历的关键词

    ###### 问答，MRC，对话，匹配，词向量，迁移，分类，分词，POS，NER等

  - 面试官就会问

    > 注意trick：方向不match的面试官喜欢考察词向量和文本分类的相关知识

    1. ##### 模型篇：**不要作死说全部熟悉，挑一两个写在熟悉算法上**

       - SGNS/cBoW、FastText、ELMo等（从词向量引出）
       - DSSM、DecAtt、ESIM等（从问答&匹配引出）
       - HAN、DPCNN等（从分类引出）
       - BiDAF、DrQA、QANet等（从MRC引出）
       - CoVe、InferSent等（从迁移引出）
       - MM、N-shortest等（从分词引出）
       - Bi-LSTM-CRF等（从NER引出）
       - LDA等主题模型（从文本表示引出）

    2. 训练篇

       - point-wise、pair-wise和list-wise（匹配、ranking模型）
       - 负采样、NCE
       - 层级softmax方法，哈夫曼树的构建
       - 不均衡问题的处理
       - KL散度与交叉熵loss函数

    3. 评价指标篇

       - F1-score
       - PPL
       - MRR、MAP

- ### 面试中的给定某业务场景的设计/方案题目

  除了基础知识，有的公司还会出一些开放性的设计题目（尤其在最后一两轮面试者为sp，ssp设置的假面时会问到）解决这些设计题主要还是靠项目和比赛经验，不要只拿论文说事

  - #### 解题核心思想

    ##### 以最小的代价来解决问题出方案，而不是非要用上最新的论文

  - #### 具体解法

    1. 能用规则解决的就不要用数据
    2. 能用简单的特征工程解决的就不要用大型神经网络
    3. 实在要上大型神经网络的时候，也尽量不要用深度LSTM这类推理复杂度太高的东西

    