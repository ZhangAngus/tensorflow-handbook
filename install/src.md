# Tensorflow源码安装

以Ubuntu 16.04为例，介绍Tensorflow源码安装的方法。

## 下载tensorflow源码

```sh
git clone https://github.com/tensorflow/tensorflow
```

## 安装bazel

```sh
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
```

## 安装依赖库

```sh
# Python 2.7
sudo apt-get install python-numpy python-dev python-pip python-wheel
# Python 3.x
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
```

## 安装CUDA和cuDNN

参考[这里](gpu.md#CUDA和cuDNN)。

## 编译安装

```sh
cd tensorflow
./configure
```

安装命令行提示，逐个设置编译选项（可以选择默认值）。

编译CPU版：

```sh
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

编译GPU版：

```sh
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
```

> 注意，GCC 5需要设置`--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`选项。

`bazel build`会生成一个`build_pip_package`命令，用来生成python whl包：

```sh
# 编译生成python whl包
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

最后，安装生成的包

```sh
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.2.0-py2-none-any.whl
```