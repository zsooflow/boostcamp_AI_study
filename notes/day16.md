#[DAY16] NLP #01

### 1. Intro to NLP, Bag-of-Words

#### A) Intro to NLP

* NLP(Natural language processing)
  * 글을 이해하는 NLU(NL **Understanding**)과 글을 생성하는 NLG(NL **Generating**)로 구성
  * Computer Vision과 더불어 인공지능과 AI가 활발하게 적용되는 분야

* Academic Disciplines

  * **NLP** (ACL, EMNLP, NAACL 등의 학회가 있음)

    * **Low-level parsing**

      ex]  Tokenization, Stemming

      : 'I STUDY MATH'라는 문장에서 각 문장의 의미를 **단어별로 쪼개서 이해**할 수 있으며, 이 단어를 **Token(토큰)**이라고 부름. 이렇게 문장을 쪼개는 것을 Tokenization이라고 부르며, 문장은 이러한 Token들이 특정 Sequence로 이루어진 것으로 파악할 수 있음.

      : Study라는 단어를 Study, Studying, Studied 등과 같이 변형할 수 있는데, 이와 같은 **수많은 어미의 변화 속에서도 이 단어들이 같은 뜻을 갖는다는 것을 파악**할 수 있어야 함. **단어의 어근만을 추출하는 것을 Stemming**이라고 함.

    * **Word and phrase level**

      ex] NER(Named Entity Recognition), POS(part-of-speech) tagging,  noun-phrase chunking, dependency parsing, coreference resolution 

      : NER같은 경우, 'New York Times'의 예를 들 수 있는데, 이러한 특성의 고유명사의 경우 단어 별로 잘라 3개의 단어로 구성되어 있다고 보면 안 되고, 이를 **하나의 고유명사로써 인식**해야 함.

      : POS tagging은, 문장 내에서 특정 단어의 품사를 뽑아내는 것.

    * **Sentence level**

      ex] Sentiment analysis, machine translation

      : 감정 분석이라 불리는 Sentiment analysis는, 특정 문장이 **긍정/부정 어조의 문장인지** 등을 파악하는 것. 'The movie was not bad.'라는 문장이 있었을 때, 'bad'라는 단어가 있더라도 문장 전체의 어조는 긍정 어조의 문장이라고 파악해야 함.

      : machine translation은, 'I love math.'라는 문장이 있을 때 이를 한글로 번역하는 것을 말함. 번역을 할 때, **문장의 어순은 물론, 단어가 문장 내에서 갖는 의미를 파악하여 적절한 번역**을 제공하는 것을 목표로 함.

    * **Multi-sentence and paragraph level**

      ex] Entailment prediction, question answering, dialog systems, summarization

      : Entailment prediction은 **문장들 사이의 모순 관계를 파악**하는 것. '어제 John이 결혼했다'와 '어제 한 명도 결혼하지 않았다'라는 문장 사이에는 모순 관계가 있다.

      : Google에서 'where did Napoleon die'라는 검색어를 입력하면 아래와 같이 그에 해당하는 답이 나온다. 이는 'where did Napoleon die'와 관련된 문서들을 쭉 나열한 뒤 독해를 통하여 **질문에 대한 답을 저절로 수행하는 과정**인데, 이를 question answering이라고 부른다.

      <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오전 10.48.31.png" alt="스크린샷 2021-02-15 오전 10.48.31" style="zoom:50%;" />

      : dialog systems는 챗봇과의 대화를 예로 들 수 있으며, summarization은 뉴스 기사를 한 줄로 요약하는 등의 예시가 있음.

  * **Text mining** (KDD, The WebConf, WSDM, CIKM, ICWSM 등의 학회가 있음)

    * **빅데이터가 활발하게 응용되는 분야**인데, 몇 십년 간의 데이터를 통하여 어떠한 트렌드를 뽑아내는 과정을 말함.

      ex] 특정 유명인과 관련된 검색어들을 추출하여 이 유명인이 어떤 이미지를 가지다가 어느 시점부터 이미지가 변화되었는지를 파악한다던가, 기업에서 어떤 제품이 출시되었을 때 제품에 대한 소비자 반응을 파악하는 것.

    * **서로 다른 키워드이지만 비슷한 의미를 가지는 키워드들을 grouping하여 분석하는 기법**을 사용해야 할 필요가 있으며, 이를 **Topic modeling(문서 군집화)**이라고 함. 

      ex] 어떤 상품에 대하여, '가성비, 내구성, 애프터 서비스' 등의 세부 내용을 말할 때 그에 대한 의견은 어떠한지 등을 자동으로 빠르게 알아볼 수 있음.

    * Social Science와 굉장히 밀접하게 연관되어 있음. Twitter를 분석하여 어떤 신조어들이 나오고 있고, 그 신조어는 어떠한 사회 현상을 반영하고 있다는 등의 연구를 수행할 수 있음.

  * **Information retrieval** (SIGIR, WSDM, CIKM, RecSys 등의 학회가 있음)
    * Google, Naver 등에서 사용하는 검색 기술을 주로 연구하는 분야. 검색 기술은 어느 정도 성숙한 수준에 머물러 있기 때문에, 기술 발전 속도는 앞의 NLP, Text mining에서보다 느린 편.
    * **추천 시스템 기술**이 여기에 속함. 영상 추천 뿐 아니라, 개인화된 광고나 상품 추천 등으로 활발하게 활용되고 있음.

* Trends of NLP

  * 2~3년 정도 전까지만 해도 Computer Vision이 새로운 구조의 CNN, GAN 등으로 빠르게 발전하고 있었던 것에 비해, 자연어 처리는 조금은 느리지만 계속하여 발전해 오고 있었음.

  * 일반적인 딥러닝 기술은 숫자로 이루어진 입/출력 데이터를 요했기 때문에, 주어진 text data를 단어 단위로 쪼개고, 이를 특정한 dimension의 vector로 표현하는 과정을 거치게 됨. 이를 **word embedding**이라고 부름. 

    ex] 'I love this movie.'를 단어 단위로 쪼개서 **문장을** **단어 단위의 sequence로 이해**할 수 있으며, 이 sequence data를 처리하는 데에 특화된 **RNN이 자연어 처리의 핵심 모델**로 자리잡게 됨. RNN에서도 **LSTMs, GRUs** 등이 많이 쓰였음.

  * 2017년도에 Google에서 학회에 'Attention Is All You Need'라는 제목의 논문을 내면서, 기존 RNN 기반의 자연어 처리 구조를 **Self-Attention 이라고 불리는 모듈으로 완전히 대체할 수 있는 transformer 모델**이 나오면서, 큰 성능 향상을 가져오게 됨. 이 task를 통하여 기존 언어학자들이 하나하나 rule을 적용하는 방식으로 **기계 번역**을 수행해왔던 것에 비해, 특별한 언어학적인 rule을 적용하지 않더라도 번역 성능이 월등하게 좋아지는 결과를 얻게 되었음. 이 transformer 모델은 기계 번역뿐만 아니라, 영상 처리, 시계열 예측, 신약/신물질 개발 등에도 활발하게 적용되어 성능 향상을 이루어내고 있음.

  * Transformer 모델이 나오기 전에는, task에 맞는 각기 다른 딥러닝 모델이 따로 존재해 왔음. self attention 모듈을 계속 쌓아나가는 식으로 모델의 크기를 키우고, 대규모 text data를 통해 이 모델을 **자가지도학습**이라는, 특정 task를 위한 별도의 label이 필요하지 않은 **범용적 task를 활용하여 모델을 학습**했음. 사전에 학습된 모델을, 큰 구조의 변화 없이도 우리가 원하는 모델으로의 **transfer learning(전위 학습)**의 형태로 적용했을 때 예전의 각기 다른 딥러닝 모델로 존재해왔던 것보다 월등한 성능 향상을 이루어 냈음.

    ex] **자가 지도 학습**은, 'I study math'라는 문장에서 study를 뺐을 때, I와 math 사이에 들어갈 수 있는 적절한 단어를 찾는 것을 말한다. 사람이라고 하더라도 정확하게 study를 뽑아낼 수는 없겠지만, 동사가 위치해야 하며, math를 목적어로 가지기에 적합한 동사가 무엇인지를 알 수 있기 때문에 이를 기계에게 수행하게 하는 것을 말한다. 이러한 자가 지도 학습의 예시로써, **BERT, GPT-3** 등의 범용 인공지능 모델이 존재함.

  * **자가 지도 학습은 대규모의 데이터 및 엄청난 양의 GPU resource를 필요**로 함. 그렇기에 최근 자연어 처리 기술 발전을 주도하는 곳은, Google, Facebook, OpenAI와 같이 막강한 자본력과 데이터가 뒷받침되는 일부 소수의 기관에서 이루어지고 있다고 한다.. ㅠㅠ

#### B) Bag-of-Words

* **Bag-of-words**

  * text-mining 분야에서 딥러닝 기술이 적용되기 이전에 많이 활용되었던 단어 및 문서를 숫자로 표현하는 bag-of-words 표현형 알아보기.

  > John really really loves this movie.
  >
  > Jane really likes this song.

1. **vocabulary 구축하기**
   
* Vocabulary : {"John", "really", "loves", "this", "movie", "Jane", "likes", "song"}
   
2. **각각의 word를 one-hot vector로 나타냄.**

   * 각각의 word는 범주형 변수로 볼 수 있고, 단어가 총 8개 있으므로 차원이 8인 one hot vector로 나타냄.

   * one-hot vector는 뒤에서 언급할 **word embedding 기법과 대비**되는 특성. 

   * John : [1 0 0 0 0 0 0 0]                     movie : [0 0 0 0 1 0 0 0]

     really : [0 1 0 0 0 0 0 0]                    Jane : [0 0 0 0 0 1 0 0]

     loves : [0 0 1 0 0 0 0 0]                    likes : [0 0 0 0 0 0 1 0] 

     this : [0 0 0 1 0 0 0 0]                       song : [0 0 0 0 0 0 0 1]

   * 유클리안 distance가 $\sqrt{2}$로 동일하며, 내적값(cosine similarity)이 0으로 동일함.

3. **sentence/document를 one-hot vector의 합으로 표현함**

   	> John really really loves this movie.

   * John + really + really + loves + this + movie : [1 2 1 1 1 0 0 0]

   > Jane really likes this song.

   * Jane + really + likes + this + song : [0 1 0 1 0 1 1 1]

* **NaiveBayes Classifier**

  * 이러한 bag of words로 나타낸 문서를 정해진 카테고리 혹은 클래스 중의 하나로 분류할 수 있는 대표적 방법

  * 문서가 분류될 수 있는 카테고리/클래스가 총 $C$개 있다고 가정하자. 수학적으로 특정 문서 $d$개가 특정 클래스 $c$에 속할 확률분포는 $P(c|d)$를 따르며, 이러한 조건부 확률분포 중 가장 높은 확률분포를 따르는 $C$ 를 택하며 문서 분류를 수행함.

  * 아래 식에서 $P(d)$ 같은 경우, 우리가 구하고자 하는 어떤 문서 d가 특정한 하나의 문서이기 때문에 상수값으로 처리가 가능하여 $\text{argmax}$ operation에서 무시 가능함
    $$
    \begin{align}
    C_{MAP} 
    &= \underset{c \in C}{\text{argmax }} P(c|d)\\
    &= \underset{c \in C}{\text{argmax }} {P(d|c)P(c) \over P(d)} \\
    &= \underset{c \in C}{\text{argmax }} P(d|c)P(c)
    \end{align}
    $$

  * 문서 $d$ 가 $w$개의 단어로 이루어졌다고 할 때, 아래와 같이 표현 가능함.

    (특정 카테고리 $c$ 안에서 각 단어가 나타날 확률이 독립이라고 가정함)
    $$
    P(d|c)P(c) = P(w_1, w_2, ... w_n|c)P(c) \to P(c) \prod_{w_i \in W} P(w_i|c)
    $$
    
* Example
  
  <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오전 11.58.18.png" alt="스크린샷 2021-02-15 오전 11.58.18" style="zoom:50%;" />
  
  1. 위의 예시에서, $P(c_{CV}) = {2 \over 4} = {1 \over 2}, P(c_{NLP}) = {2 \over 4} = {1 \over 2}$ 이다.
  
  2. CV Class 내 단어 개수는 총 14개인데 $P(w_{task}|c_{CV}) = {1 \over 14}$ 로, 마찬가지로 NLP Class 내 단어 개수가 총 10개 있으므로 $P(w_{task}|c_{CV}) = {2 \over 10}$ 과 같은 형태로 표현 가능하다.
  
  3. 각 class 별로 분류하고자 하는 text document의 확률 분포를 계산한다.
       $$
       P(c_{CV} |d_5) = P(c_{CV}) \prod_{w \in W} P(w|c_{CV}) = {1\over 2} \times {1 \over 14} \times {1\over 14} \times {1\over 14} \times {1\over 14}\\P(c_{NLP} |d_5) = P(c_{NLP}) \prod_{w \in W} P(w|c_{NLP}) = {1\over 2} \times {1 \over 10} \times {2\over 10} \times {1\over 10} \times {1\over 10}
       $$
  
* Traning Data 내에 Test Data에 나온 단어가 한 번도 나오지 않았다면 그 확률은 0으로 추정될 것이며, 다른 단어들이 아무리 해당 class와 밀접한 관련이 있다고 하더라도 무조건 0으로 추정되게 됨. 이 때문에 다양한 regularization이 함께 추가되어 활용되곤 함.

### 2. Word Embedding

* **Word Embedding**
  * 각 단어들을 특정 공간의 한 점의 벡터로 나타내는 것
  * 'cat'과 'kitty'는 유사한 단어이기 때문에 비슷한 vector로 표현이 되어 거리가 짧고, 'hamberger'와는 유사하지 않은 단어이기 때문에 다른 vector로 표현이 되어 서로 간의 거리가 길게 나타난다. 
  * text data set을 학습 데이터로 주고 좌표 공간의 dimension 수를 사전에 정의하여 주면, 학습이 완료된 이후 공간 내에서 최적의 좌표 값들을 output으로 주게 됨.
  * **단어 상의 유사도를 잘 반영**함.

#### A) Word2Vec

* **특정 단어로부터 문장에서 비슷한 위치에 있는 단어들은 비슷한 의미를 띨 것이라는 가정**을 중심으로 학습을 진행하게 됨.

  > The **cat** purrs.
  >
  > This **cat** hunts mice.

  * 위의 두 문장에서, 'cat'이라는 단어를 중심으로 'The', 'This', 'purrs', 'hunts', 'mice' 의 단어들이 모두 유사성이 있다고 가정함.

* 어떤 문장에서 특정 단어 (ex.'cat')이 나왔을 때, **그 특정 단어 주위의 단어들을 지우고 난 뒤 특정 단어 주위에 어떤 단어들이 올지를** 학습함.

  <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오후 2.22.15.png" alt="스크린샷 2021-02-15 오후 2.22.15" style="zoom:50%;" />

* **How works**

  > I study math.

  1. **Tokenization**

     : 각 단어 별로 쪼갬

  2. **Vocabulary 생성**

     : {"I", "study", "math"}

  3. **one-hot vector 표현**

     : vocabulary 사이즈만큼의 dimension을 가지는 one-hot vector로 표현함

     : "I" [1 0 0] "study" [0 1 0] "math" [0 1 1]

  4. **sliding window 기법 적용**

     : window의 사이즈가 3일 경우(특정 단어를 중심으로 앞 뒤로 하나씩 볼 경우), "I"라는 단어를 중심으로 한다면, (I, study)라는 Input, Output 쌍이 만들어짐. 이 때 **sliding** window이기 때문에 한 칸 옮겨가서, "study"라는 단어를 중심으로 (study, I), (study, math) 라는 입출력 쌍이 만들어짐.

  5. **softmax 계산**

     : Input과 Output의 dimension이 3이기 때문에 각각의 Input/Output layer 수는 3이 되며, hidden layer의 node 수는 사용자가 정하는 hyper parameter로써, word embedding을 수행하는 좌표 공간의 차원 수와 동일하게 설정함. 

     : 위에서 (study, math) 쌍을 중심으로 예시를 보면 아래와 같다

     <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오후 2.35.55.png" alt="스크린샷 2021-02-15 오후 2.35.55" style="zoom:50%;" />

     * hidden layer를 2로 설정하였을 때, 위의 빨간 박스에서와 같이 3 2 3의 형태가 되며, $W_1$ 은 2x3 matrix, $W_2$ 는 3x2 matrix로 구성된다.
     * $W_1$ 과 $x$ 가 곱해지는 것을 **embedding layer**라고 부르며, 행렬곱 연산 대신 $x$ 자리에 있는 값만을 가져오는 식으로 연산이 수행된다.
     * softmax의 결과가 우리의 [ground Truth](https://eair.tistory.com/16) 결과와 유사하게 됨.

* **시각화 Example**

  * [본 사이트](https://ronxin.github.io/wevi/)에 들어가게 되면, 아래와 같은 모습이 나오는데, input/output size가 8이고 hidden layer가 5로 설정되어 있다.

    <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오후 2.54.07.png" alt="스크린샷 2021-02-15 오후 2.54.07" style="zoom:50%;" />

  * 붉은 색은 양수, 푸른색은 음수 값으로 되어 있는데, weight matrix가 아래와 같이 학습이 되면서 weight가 조절된다.

    <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오후 2.55.01.png" alt="스크린샷 2021-02-15 오후 2.55.01" style="zoom: 25%;" /><img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오후 2.55.07.png" alt="스크린샷 2021-02-15 오후 2.55.07" style="zoom: 25%;" />

  * 이후 나온 결과값을 보면, 처음 training data에 넣었던 값과 유사하게, juice, milk, water의 input 값이 drink의 output과 유사해졌음을 파악할 수 있다.

    <img src="/Users/jisukim/Desktop/와아ㅏ.png" alt="와아ㅏ" style="zoom:50%;" />

* vector relationship

  <img src="/Users/jisukim/Desktop/스크린샷 2021-02-15 오후 2.59.43.png" alt="스크린샷 2021-02-15 오후 2.59.43" style="zoom:50%;" />

  * 위에서 vec[queen] - vec[king] = vec[woman] - vec[man]의 관계를 통하여 남성과 여성 간 단어의 거리를 잘 구별했음을 알 수 있다.

* another example

  * http://w.elnn.kr/search  (한국-서울)+도쿄 = 일본

  * [word intrusion detection](https://github.com/dhammack/Word2VecExample) : 문장의 단어 중, 가장 의미가 다른 것을 고르는 것

    ex. math **shopping** reading science

#### B) GloVe

* 각 입출력 쌍에 대해서 **학습 데이터에서 두 단어가 한 window 내에서 몇 번 등장했는지 사전에 계산을 한다는 것**이 Word2Vec과 가장 큰 차이.
  $$
  J(\theta) = {1 \over 2} \sum_{i,j = 1}^W f(P_{ij})(u_i^Tv_j - log{P_{ij}})^2
  $$

* Word2Vec같은 경우 학습이 빈번하게 될 수록 그 내적값이 커지는 학습방식을 따르고 있는데, GloVe는 특정 단어 쌍이 동시에 등장한 횟수를 미리 계산하고 그 값에 로그를 취하여 두 단어값의 ground truth로 사용함으로써 중복계산을 줄여주었다.
* **학습 속도가 빠르고, 적은 데이터에 대해서 학습을 잘 한다는 장점이 있다.**

* 추천 시스템에 많이 사용되는 알고리즘으로 이해할 수도 있음.









