# 文件构成

- error.cuh：CUDA API错误诊断使用的宏定义头文件，用于诊断CUDA API的返回错误，显示错误发生的具体文件和行数，便于调试。
- GeneticAlgorithm.hpp：遗传算法通用类模板和函数模板的头文件，包含了所有实际应用场景的遗传算法的通用特性
- image_helpers.hpp ：图片处理工具包
- crossover.hpp：定义交叉操作
- puzzle.cu ：主程序文件

# 环境

- 操作系统：Ubuntu18.04LTS
- 软件工具：cuda-toolkits 10.2 , cmake 

# 编译、运行、调试指令

## 编译

```sh
$ cmake .
$ make
```

## 运行

```sh
$ ./puzzle
```

## 调试

```sh
$ cuda-gdb ./puzzle
$ run
```

