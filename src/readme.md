# 文件构成

- error.cuh：CUDA API错误诊断使用的宏定义头文件，用于诊断CUDA API的返回错误，显示错误发生的具体文件和行数，便于调试。
- GeneticAlgorithm.hpp：遗传算法通用类模板和函数模板的头文件，包含了所有实际应用场景的遗传算法的通用特性
- GA1.cu ：实例1 求使目标函数最优值的参数值
- GA2.cu ：实例2 学习感知机模型根据身高体重预测性别

# 环境

- 操作系统：Ubuntu18.04LTS
- 软件工具：cuda-toolkits 10.2

# 编译、运行、调试指令

以实例1为例

## 编译

```sh
$ nvcc -o GA -g -G GA1.cu
```

## 运行

```sh
$ ./GA1
```

## 调试

```sh
$ cuda-gdb ./GA1
$ run
```

