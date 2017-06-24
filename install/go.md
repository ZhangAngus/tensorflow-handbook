# Tensorflow For Go

Tensorflow For Go支持Linux和OSX。

## 安装

### 下载动态链接库

```sh
$ TF_TYPE="cpu" # Change to "gpu" for GPU support
$ TARGET_DIRECTORY='/usr/local'
$ curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.2.0.tar.gz" | sudo tar -C $TARGET_DIRECTORY -xz
# Linux上还需要执行ldconfig
$ sudo ldconfig
```

### 下载Tensorflow Go库

```sh
$ go get github.com/tensorflow/tensorflow/tensorflow/go
```

下载完成后，可以用`go test`验证安装

```sh
$ go test github.com/tensorflow/tensorflow/tensorflow/go
```

## Go示例

```sh
$ cat <<EOF | tee tf-hello.go
package main

import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "github.com/tensorflow/tensorflow/tensorflow/go/op"
    "fmt"
)

func main() {
    // Construct a graph with an operation that produces a string constant.
    s := op.NewScope()
    c := op.Const(s, "Hello from TensorFlow version " + tf.Version())
    graph, err := s.Finalize()
    if err != nil {
        panic(err)
    }

    // Execute the graph in a session.
    sess, err := tf.NewSession(graph, nil)
    if err != nil {
        panic(err)
    }
    output, err := sess.Run(nil, []tf.Output{c}, nil)
    if err != nil {
        panic(err)
    }
    fmt.Println(output[0].Value())
}
EOF

$ export LIBRARY_PATH=/usr/local/lib
$ go run tf-hello.go
2017-06-19 15:38:38.569010: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-19 15:38:38.569633: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-06-19 15:38:38.569640: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-19 15:38:38.569646: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Hello from TensorFlow version 1.2.0
```

## C示例

C和Go实际上共用了相同的动态链接库和头文件，安装好Go版本后，C的API可以直接使用，如

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    return 0;
}
```

编译，运行

```sh
$ gcc -I/usr/local/include -L/usr/local/lib tf.c -ltensorflow
$ ./a.out
Hello from TensorFlow C library version 1.2.0
```
