# Tensorflow CPU版本安装

## pip安装

最简单的方法使用pip来安装

```sh
# Python 2.7
pip install --upgrade tensorflow
# Python 3.x
pip3 install --upgrade tensorflow
```

## docker

使用镜像`gcr.io/tensorflow/tensorflow`启动CPU版Tensorflow：

```sh
docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```

## 验证安装

```sh
$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>>
```
