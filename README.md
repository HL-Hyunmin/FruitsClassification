# Fruit classification using deep learning
20145156  이현민

---
# 프로젝트 목적
 - 과일이미지에 대해 과일을 분류하는 모델을 설계하여 해당 이미지에 맞는 과일을 정확하게 분류하는 것을 목적으로 하고 있고 차후에는 모델을 더 개선하여 실 생활에서 쓰일 수 있는 실제 과일에 대해 분류하여 해당하는 가격을 보여주는 등 상용화 되어 질 수 있다고 기대된다.  

* 기존에 딥러닝 모델을 구성할 때처럼  어느 정도 감에 의존하며 모델을 개선을 하는 것이 아니라 다양한 지표를 통해 모델이 학습하는데 어느 부분(저차원, 고차원)이 부족했는지 등등 모델을 근거 있게 개선하는 것 또한 목표로 하고 있다.

# 프로그램 개요
 - 과일을 분류하는 초기 모델을 설계하고 초기 모델에 대한 그래프와 학습 결과를 바탕으로 어느 부분(저차원 , 고차원인지)이 부족한지 완벽하게 예측 할 수는 없지만  해당 부분에 대해 의심을 가지고 모델을 재구성한다.

# 프로그램 구성
- Train Model -> Plotting -> Test Model -> Feed Back -> Train Model_improved<br> -> Plotting ->  Test Model

# Image Data
- Kaggle에서 제공하는 fruit-360 데이터 셋 사용.
초기 이미지는 100 by 100 형태의 컬러 이미지 이고 이미지는 과일에 대해 360도로 나누어 부분 부분이 저장되어 있는 형태이다.<br>
Train set `53177`, Test set  `17835` 장으로 설정하였고 과일은 103종의 과일을 분류하였다.

# Train Model
  - 데이터 셋 코드 설명 및 초기 모델에 대한 전반적인 설명

```python
batch_size = 350
num_epoch = 150
keep_prob = 0
np.random.seed(5000)

def load_dataset(isTrain=True):
    if isTrain:
        path = './fruits-360/Training'
    else: path = './fruits-360/Test'
    image_list = []
    label_list = []

    folder_list = os.listdir(path)

    for folder in folder_list:
        p = os.path.join(path, folder)
        image_ = os.listdir(p)
        label = folder.split("[")[1].split("]")[0]
        for image in image_:
            image1 = Image.open(os.path.join(p, image))
            image1 = image1.resize((100,100))
            image_list.append(np.array(image1))
            label_list.append(np.int(label))

    image_list = np.array(image_list)
    label_list =np.array(label_list)

    image_list, label_list = shuffle_(image_list, label_list)

    return image_list, label_list

def shuffle_(X, y):
    R = np.random.permutation(len(X))
    return X[R], y[R]
```
 위 코드는
Kaggle에서 가져온 데이터들을 목적에 맞게 변형한 뒤 불러오는 역할을 한다.<br>
편중되서 학습되는 것을 방지하기 위해 데이터들을 shuffle 하는 함수를 작성하였다.
<br><br>

```python
def model(X):
    net = tf.cast(X, tf.float32)
    net = net / 255.0


    net = slim.conv2d(net, 32, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 64, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 128, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 256, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 512, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.flatten(net)
    net = tf.nn.dropout(net, 0.6)
    net = slim.fully_connected(net, 103)

    return net
```
초기 모델은 100 by 100의 이미지를 5번의 Convolution, max_pooling 하여서  
최종적으로 4 by 4 의 크기를 가지고 1차원 형태로 변형해서 최종적으로 클래스 갯수(103)에 맞게 모델을 만들었다.
Convolution layer에 초기 부분은 저차원(색, 작은특징),  진행 될 수록 고차원(외형, 큰 특징)을 의미한다.
<br><br>
```python
if __name__ == "__main__":

    train_x, train_y = load_dataset()
    test_x, test_y = load_dataset(False)

    np.mean(train_x), np.std(train_x))

    X = tf.placeholder(tf.uint8, [None, 100, 100, 3], name='input')
    y = tf.placeholder(tf.int64, [None], name='y')

    logits = model(X)
```
메인문으로써 Train, Test 셋을 불러오고 변수들을 설정하는 코드이다.
<br><br>
```python
    with tf.name_scope("optimizer"):
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.AdadeltaOptimizer(0.07).minimize(cost)
```
cost 함수로 cross_entropy를 사용하였고 최적화 함수로 Adadelta를 사용하였다
<br><br>

```python
    y_hat = tf.argmax(logits, -1)
    correct_prediction = tf.equal(y_hat, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    total_batch  = int(len(train_x) / batch_size)
```

후에 argmax 와 정확도를 확인하기 위해 변수를 설정하는 코드이다. total_batch 는 1epoch 에 학습 하는 수를 나타내고 있다.
<br><br>

```pyhton
    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model restored.')
    else:
        sess.run(tf.global_variables_initializer())
    print('Learning Start')
```

모델 그래프를 실행하기 위한 Session 설정

모델 실행 중 중간에 종료하고 재실행하여도 기존에 학습하던게 남아있어 학습되었던 가중치가 유지되어 재실행된다.
<br><br>
```python
    for epoch in range(num_epoch):
        avg_cost = 0
        avg_accuracy = 0

        f = open("log.txt", 'a')

        for idx in range(total_batch):
            start = ((idx) * batch_size)
            end = ((idx+1) * batch_size)
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]

            feed_dict1 = {X:batch_x, y:batch_y}

            _y_hat, cost_, _ , accuracy_ = sess.run([y_hat, cost, optimizer, accuracy], feed_dict=feed_dict1)

            avg_cost += cost_ / total_batch
            avg_accuracy += accuracy_ / total_batch

        avg_cost_test = 0
        avg_accu_test = 0
        total_test_batch = int(len(test_x) / batch_size)
```
train set에 대해 model을 fitting 하고 해당하는 인자에 맞게 대입해주고 모델 중간 중간 진행 과정을 보기 위해 avg_cost,accuracy를 설정한다.
<br><br>

```python
        for idx in range(total_test_batch):
            start = idx * batch_size
            end = (idx+1) * batch_size
            batch_x = test_x[start:end]
            batch_y = test_y[start:end]
            feed_dict2 = {X: batch_x, y: batch_y}

            _y_hat_test, cost_t, accu_t = sess.run([y_hat, cost, accuracy], feed_dict=feed_dict2)
            avg_cost_test += cost_t / total_test_batch
            avg_accu_test += accu_t / total_test_batch
```
Train set fitting 과 동일한 역할을 한다 Test set 이기때문에 optimizer는 제외 된 걸 확인 할 수있다.
<br><br>

```python
        if epoch % 2 ==0:
            print("Training Cost: %.5f Accuracy: %.5f (test cost: %.5f accuracy: %.5f " % (avg_cost, avg_accuracy, avg_cost_test, avg_accu_test))
            f.write(str(epoch)+" epoch "+str(avg_cost)+" "+str(avg_accuracy)+" "+str(avg_cost_test)+" "+str(avg_accu_test)+"\n")
            saver.save(sess, os.path.join("./model", 'model.ckpt'))

        f.close()
```
모델이 학습 되어지는 중간 중간 진행 상황을 보기 위해 학습과정을 뽑아내는 코드이다.

# Plotting

- 원하는 모델에 대한 Plotting 하는 코드이다. epochs(x축)에 대한cost, accuracy(y축)를 나타내는 그래프를 만들었다.<br><br>

```python
f = open("log.txt", 'r')

lis = f.readlines()


epo = []

training_cost = []
test_cost = []

training_accu = []
test_accu = []

all_length = len(lis)
for line in lis:
    a = line.split(" ")
    epo.append(int(a[0]))
    training_cost.append(float(a[2]))
    training_accu.append(float(a[3]))
    test_cost.append(float(a[4]))
    test_accu.append(float(a[5]))


plt.plot(epo, training_cost, '-', color='blue')
plt.plot(epo, test_cost, '-', color='red')
plt.xlabel("Epoch")
plt.ylabel("Cost")
fig = plt.gcf()
fig.savefig("Cost.jpg")
```

train, test 에 대한 cost,accuracy 를 리스트에 저장하여 한 그래프에 train,test 에 대한 cost 혹은 accuracy를 나타낸다.

![base_acc](C:/Users/hwlee/Desktop/DM_Project_img/base_acc.png)


# Test Model
- 학습한 모델에 대한 session을 재 설정하고 Confusion Maxtrix 그래프를 그린다.

```python
data, label = load_dataset(False)

X = tf.placeholder(tf.uint8, [None, 100, 100, 3], name='input')
y = tf.placeholder(tf.int64, [None], name='y')

logits = model(X)
sess = tf.Session()
save = tf.train.Saver()
try:
    save.restore(sess, save_path='./model/model.ckpt')
except:
    print("Load Failure")
    exit()

batch_size = 500
total_batch = int(len(data) / batch_size)
fy_list = []

y_hat = tf.argmax(logits, -1)
correct_prediction = tf.equal(y_hat, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for idx in range(total_batch):
    start = idx * batch_size
    end = (idx+1) * batch_size

    batch_x = data[start:end]
    feed_dict = {X: batch_x}

    _y_hat = sess.run(y_hat, feed_dict=feed_dict)

    fy_list.extend(_y_hat)

for i in range(len(fy_list)):
    if fy_list[i] != label[i]:
        Image.fromarray(data[i]).save(os.path.join('./FalseImage',str(i)+":pred_"+str(fy_list[i])+"true_"+str(label[i])+'.jpg'))
```
Model에 대한 Session을 재설정한다.<br><br>

```python
save_confusion_matrix(label[:len(fy_list)], fy_list, label=list(range(103)),path='confusion_matrix.png')
```
학습이 완료된 모델에 대해 모델이 예측한 값에 대해 실제 라벨 값과 다르게 예측한 것을 나타내는 지표인 Confusion Matrix 그래프를 그린다.

![base_acc](C:/Users/hwlee/Desktop/DM_Project_img/base_confusion.png )

# Feed Back
- 초기 모델에 대한 그래프와 학습 결과를 바탕으로 어느 부분(저차원 , 고차원인지)이 부족한지 완벽하게 예측 할 수는 없지만  해당 부분에 대해 의심을 가지고 모델을 재구성한다.

# Train Model_improved
- 각 종 지표를 확인해보았을 때 저차원 특징이 부족하고 전체적으로 overfitting 되었다고 의심이 들어서 이에 대한 보완으로 저차원 layer 특징 수와 Dropout layer를 추가하고 Hidden Layer 수는 줄였다.

- 일부의 특징 만 가지고 모델을 판단 할 수 있으므로 L2 loss를 추가하였다.

```python
batch_size = 350
num_epoch = 150
keep_prob = 0
np.random.seed(5000)
def load_dataset(isTrain=True):
    if isTrain:
        path = './fruits-360/Training'
    else: path = './fruits-360/Test'

    image_list = []
    label_list = []

    folder_list = os.listdir(path)

    for folder in folder_list:
        p = os.path.join(path, folder)
        image_ = os.listdir(p)
        label = folder.split("[")[1].split("]")[0]
        for image in image_:
            image1 = Image.open(os.path.join(p, image))
            image1 = image1.resize((96,96)) # Revised 1
            image_list.append(np.array(image1))
            label_list.append(np.int(label))


    image_list = np.array(image_list)
    label_list =np.array(label_list)

    image_list, label_list = shuffle_(image_list, label_list)

    return image_list, label_list
```
Img Size 를 96 by 96으로 조정하였다.<br><br>

```pyhton
def shuffle_(X, y):
    R = np.random.permutation(len(X))
    return X[R], y[R]

def model(X):
    net = tf.cast(X, tf.float32)
    net = net / 255.0


    net = slim.conv2d(net, 64, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 32, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 128, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = slim.conv2d(net, 256, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))
    net = tf.nn.dropout(net, 0.6)
    net = slim.conv2d(net, 512, kernel_size = (3, 3))
    net = slim.max_pool2d(net, (2, 2))

    net = slim.flatten(net)
    net = tf.nn.dropout(net, 0.6)
    net = slim.fully_connected(net, 103)

    return net
```
저차원 layer 의 특징 수를 늘리고 Dropout layer를 추가하였다.
<br>
```python
if __name__ == "__main__":


    train_x, train_y = load_dataset()
    test_x, test_y = load_dataset(False)

    np.std(train_x))

    X = tf.placeholder(tf.uint8, [None, 96, 96, 3], name='input')
    y = tf.placeholder(tf.int64, [None], name='y')

    logits = model(X)


    with tf.name_scope("optimizer"):
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.001
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)) + lossL2
        optimizer = tf.train.AdadeltaOptimizer(0.07).minimize(cost)
```

일부 특징으로 결과를 단정하는 것을 방지하기 위해 여러 특징들에 가중치를 적당하게 사용한다.
<br>
```python
    y_hat = tf.argmax(logits, -1)
    correct_prediction = tf.equal(y_hat, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    total_batch  = int(len(train_x) / batch_size)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model_improved')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model restored.')
    else:
        sess.run(tf.global_variables_initializer())
    print('Learning Start')


    for epoch in range(num_epoch):
        avg_cost = 0
        avg_accuracy = 0

        f = open("log_improved.txt", 'a')
        for idx in range(total_batch):
            start = ((idx) * batch_size)
            end = ((idx+1) * batch_size)
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]

            feed_dict1 = {X:batch_x, y:batch_y}

            _y_hat, cost_, _ , accuracy_ = sess.run([y_hat, cost, optimizer, accuracy], feed_dict=feed_dict1)

            avg_cost += cost_ / total_batch
            avg_accuracy += accuracy_ / total_batch

        avg_cost_test = 0
        avg_accu_test = 0
        total_test_batch = int(len(test_x) / batch_size)

        for idx in range(total_test_batch):
            start = idx * batch_size
            end = (idx+1) * batch_size
            batch_x = test_x[start:end]
            batch_y = test_y[start:end]
            feed_dict2 = {X: batch_x, y: batch_y}
            _y_hat_test, cost_t, accu_t = sess.run([y_hat, cost, accuracy], feed_dict=feed_dict2)
            avg_cost_test += cost_t / total_test_batch
            avg_accu_test += accu_t / total_test_batch


        if epoch % 2 ==0:
            print("Training Cost: %.5f Accuracy: %.5f (test cost: %.5f accuracy: %.5f " % (avg_cost, avg_accuracy, avg_cost_test, avg_accu_test))
            f.write(str(epoch)+" epoch "+str(avg_cost)+" "+str(avg_accuracy)+" "+str(avg_cost_test)+" "+str(avg_accu_test)+"\n")
            saver.save(sess, os.path.join("./model_improved", 'model.ckpt'))

        f.close()

```
초기 모델 코드와 동일한 역할을 한다.
![base_acc](C:/Users/hwlee/Desktop/DM_Project_img/new_cost.png)
# Plotting & Test
 - 이하 초기 모델의 Plotting, Test Model 과 동일하다.
![base_acc](C:/Users/hwlee/Desktop/DM_Project_img/new_confusion.png)

# 결론 및 느낀 점
- 초기 모델을 개선 해서 초기 보다 적합한 모델을 만들었지만 개선 모델의 Confusion Metrix로 보아 아직 overfitting 되었을 가능성이 있고, 충분히 더 개선 될 수 있다.

- 분류 모델을 설계할 때 감에 의존해서 막연하게 모델을 구현, 실행 해보는 것이 아닌 효율적으로 설계하는 법을 배웠다.
