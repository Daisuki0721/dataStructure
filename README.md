# 图像分割工具 - C++ Qt GUI 版本

## 功能概述
本项目是一个基于 C++ 和 Qt5 的图形化界面应用，用于图像分割。支持：
- 上传本地图像文件
- 自定义 K 值（种子点数量）
- 实时查看原始图像和分割结果
- 保存分割结果

## 系统需求
- macOS 10.12+ 或 Linux
- C++ 17 标准编译器
- CMake 3.10+
- OpenCV 4.5+
- Qt5 5.15+

## 安装步骤

### 1. 安装依赖（macOS）

```bash
# 使用 Homebrew 安装依赖
brew install opencv qt5 cmake
```

### 2. 编译项目

```bash
# 进入项目文件夹
cd dataStructure

# 创建 build 目录
mkdir -p build

# 配置并编译
cmake -B build
cd build
make
```

### 3. 运行应用

```bash
# 运行 GUI 版本
./task_gui

# 运行命令行版本
./task
```

## 使用说明

### 启动应用
运行 `./task_gui` 后，会弹出 GUI 窗口。

### 1.选择图像
- 点击 **"浏览文件"** 按钮
- 选择要处理的图像文件（支持 JPG、PNG、BMP、TIFF 等）
- 选定后，原始图像会显示在 **"原始图像"** 标签页中

### 2.设置参数
- **K 值**：设置种子点数量
- 使用微调框选择合适的 K 值

### 3.执行分割
- 点击 **"开始分割"** 按钮
- 应用执行以下步骤：
  1. 生成均匀分布的种子点
  2. 执行分水岭分割算法
  3. 生成并显示结果
- 进度会在状态栏实时显示

### 步骤 4：查看结果
- 状态栏显示处理时间和种子点数量
- 红色圆点为原始种子点位置

### 步骤 5：保存结果
- 点击 **"保存结果"** 按钮
- 选择保存位置和格式（PNG 或 JPG）
- 分割结果保存为指定文件

## 项目结构

```
dataStructure/
├── assets
│   ├── baboon.jpg
│   ├── fruits.jpg
│   └── lena.jpg
├── CMakeLists.txt
├── include/
│   ├── segmentation.hpp
│   └── image_gui.hpp
├── src/
│   ├── main.cpp
│   ├── main_gui.cpp
│   └── image_gui.cpp
├── build/
└── README.md
```
