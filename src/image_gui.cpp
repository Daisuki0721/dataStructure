#include "image_gui.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollArea>
#include <QTabWidget>
#include <QPixmap>
#include <QImage>
#include <QApplication>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "segmentation.hpp"

using namespace cv;
using namespace std;

ImageGUI::ImageGUI(QWidget *parent)
    : QMainWindow(parent), original_image(), result_image(), k_value(25), tab_widget(nullptr)
{
    initUI();
}

ImageGUI::~ImageGUI() {}

void ImageGUI::initUI()
{
    setWindowTitle("图像分割工具 - Watershed Algorithm");
    setGeometry(100, 100, 1400, 900);

    // 创建中心widget
    QWidget *central_widget = new QWidget(this);
    setCentralWidget(central_widget);

    // 主布局
    QVBoxLayout *main_layout = new QVBoxLayout(central_widget);

    // 控制面板
    QWidget *control_panel = createControlPanel();
    main_layout->addWidget(control_panel);

    // Tab用于显示图像
    tab_widget = new QTabWidget();

    // 原始图像
    original_label = new QLabel();
    original_label->setStyleSheet("border: 1px solid #ccc;");
    original_label->setMinimumHeight(400);
    original_label->setMinimumWidth(600);
    original_label->setAlignment(Qt::AlignCenter);
    original_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QScrollArea *original_scroll = new QScrollArea();
    original_scroll->setWidget(original_label);
        original_scroll->setWidgetResizable(true);
    tab_widget->addTab(original_scroll, "原始图像");

    // 分割结果
    result_label = new QLabel();
    result_label->setStyleSheet("border: 1px solid #ccc;");
    result_label->setMinimumHeight(400);
    result_label->setMinimumWidth(600);
    result_label->setAlignment(Qt::AlignCenter);
    result_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QScrollArea *result_scroll = new QScrollArea();
    result_scroll->setWidget(result_label);
        result_scroll->setWidgetResizable(true);
    tab_widget->addTab(result_scroll, "分割结果");

    main_layout->addWidget(tab_widget);

    // 状态信息
    status_label = new QLabel("准备就绪");
    status_label->setStyleSheet("color: #666; padding: 10px;");
    main_layout->addWidget(status_label);

    show();
}

QWidget *ImageGUI::createControlPanel()
{
    QWidget *panel = new QWidget();
    QHBoxLayout *layout = new QHBoxLayout(panel);

    // 上传图片
    layout->addWidget(new QLabel("选择图片:"));
    file_button = new QPushButton("浏览文件");
    connect(file_button, &QPushButton::clicked, this, &ImageGUI::selectImage);
    layout->addWidget(file_button);

    file_label = new QLabel("未选择文件");
    file_label->setStyleSheet("color: #999;");
    layout->addWidget(file_label);

    layout->addSpacing(20);

    // K值输入
    layout->addWidget(new QLabel("K值 (种子数):"));
    k_spinbox = new QSpinBox();
    k_spinbox->setMinimum(1);
    k_spinbox->setMaximum(1000);
    k_spinbox->setValue(25);
    connect(k_spinbox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ImageGUI::onKValueChanged);
    layout->addWidget(k_spinbox);

    layout->addSpacing(20);

    // 开始分割按钮
    segment_button = new QPushButton("开始分割");
    connect(segment_button, &QPushButton::clicked, this, &ImageGUI::startSegmentation);
    segment_button->setStyleSheet(
        "QPushButton {"
        "    background-color: #4CAF50;"
        "    color: white;"
        "    border: none;"
        "    padding: 8px 20px;"
        "    border-radius: 4px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #45a049;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #3d8b40;"
        "}");
    layout->addWidget(segment_button);

    layout->addStretch();

    // 保存结果按钮
    save_button = new QPushButton("保存结果");
    connect(save_button, &QPushButton::clicked, this, &ImageGUI::saveResult);
    save_button->setEnabled(false);
    save_button->setStyleSheet(
        "QPushButton {"
        "    background-color: #2196F3;"
        "    color: white;"
        "    border: none;"
        "    padding: 8px 20px;"
        "    border-radius: 4px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #0b7dda;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #0a66c2;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #ccc;"
        "}");
    layout->addWidget(save_button);

    return panel;
}

void ImageGUI::selectImage()
{
    QString file_path = QFileDialog::getOpenFileName(
        this,
        "选择图像文件",
        "",
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)");

    if (!file_path.isEmpty()) {
        image_path = file_path.toStdString();
        file_label->setText(QFileInfo(file_path).fileName());
        file_label->setStyleSheet("color: #333; font-weight: bold;");

        // 读取并显示原始图像
        original_image = imread(image_path);
        if (!original_image.empty()) {
            displayImage(original_image, original_label);
            tab_widget->setCurrentIndex(0);
            string status = "已加载图像: " + to_string(original_image.cols) + "x" +
                          to_string(original_image.rows);
            status_label->setText(QString::fromStdString(status));
        } else {
            QMessageBox::warning(this, "错误", "无法读取图像");
            status_label->setText("读取图像失败");
        }
    }
}

void ImageGUI::onKValueChanged(int value)
{
    k_value = value;
}

void ImageGUI::startSegmentation()
{
    if (image_path.empty()) {
        QMessageBox::warning(this, "警告", "请先选择图像文件");
        return;
    }

    if (original_image.empty()) {
        QMessageBox::warning(this, "警告", "图像为空");
        return;
    }

    // 禁用按钮
    segment_button->setEnabled(false);
    file_button->setEnabled(false);
    k_spinbox->setEnabled(false);

    status_label->setText("正在处理");

    try {
        // 开始计时
        auto start_time = chrono::high_resolution_clock::now();

        status_label->setText("生成种子点");
        qApp->processEvents();

        int width = original_image.cols;
        int height = original_image.rows;

        // 生成均匀随机种子点
        vector<Point> seedPoints = SeedSampler::generateSeeds(width, height, k_value);

        Mat markers = Mat::zeros(original_image.size(), CV_32S);
        for (size_t i = 0; i < seedPoints.size(); i++) {
            circle(markers, seedPoints[i], 1, Scalar(static_cast<int>(i + 1)), -1);
        }

        // 预处理：均值漂移滤波
        Mat shifted;
        pyrMeanShiftFiltering(original_image, shifted, 10, 51);

        status_label->setText("执行分水岭算法");
        qApp->processEvents();

        // 分水岭
        watershed(shifted, markers);

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end_time - start_time;

        status_label->setText("生成标记");
        qApp->processEvents();

        // 生成随机颜色
        vector<Vec3b> colors;
        srand(static_cast<unsigned int>(time(nullptr)));
        for (int i = 0; i < k_value; i++) {
            colors.push_back(Vec3b(rand() & 255, rand() & 255, rand() & 255));
        }

        // 创建彩色mask
        Mat colorMask = Mat::zeros(original_image.size(), CV_8UC3);
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int index = markers.at<int>(i, j);
                if (index > 0 && index <= k_value) {
                    colorMask.at<Vec3b>(i, j) = colors[index - 1];
                } else if (index == -1) {
                    // 分水岭边界显式着色，避免视觉上出现“空线”
                    colorMask.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
                }
            }
        }

        // 合成结果
        Mat result;
        addWeighted(original_image, 0.6, colorMask, 0.4, 0, result);

        // 绘制种子点
        for (const auto &pt : seedPoints) {
            circle(result, pt, 3, Scalar(0, 0, 255), -1);
            circle(result, pt, 3, Scalar(255, 255, 255), 1);
        }

        // 添加时间信息
        string timeText = "Time: " + to_string(duration.count()) + " ms";
        putText(result, timeText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 255, 0), 2);

        result_image = result;
        displayImage(result, result_label);
        tab_widget->setCurrentIndex(1);

        status_label->setText(QString("完成! 执行耗时: %1 ms, 生成种子点: %2")
                                 .arg(duration.count(), 0, 'f', 2)
                                 .arg(seedPoints.size()));

        save_button->setEnabled(true);

    } catch (const exception &e) {
        QMessageBox::critical(this, "错误", QString("处理失败: %1").arg(e.what()));
        status_label->setText("处理失败");
    }

    // 启用按钮
    segment_button->setEnabled(true);
    file_button->setEnabled(true);
    k_spinbox->setEnabled(true);
}

void ImageGUI::displayImage(const Mat &cv_img, QLabel *label)
{
    Mat display_img = cv_img.clone();

    // 自动调整大小
    int height = display_img.rows;
    int width = display_img.cols;
    if (width > 800 || height > 500) {
        double scale = min(800.0 / width, 500.0 / height);
        int new_width = static_cast<int>(width * scale);
        int new_height = static_cast<int>(height * scale);
        cv::resize(display_img, display_img, Size(new_width, new_height));
    }

    // 转换颜色空间
    Mat rgb_image;
    cvtColor(display_img, rgb_image, COLOR_BGR2RGB);

    // 按 OpenCV 行步长构建 QImage，并 copy() 一份独立内存，避免显示空行/条纹
    QImage qt_image(rgb_image.data,
                    rgb_image.cols,
                    rgb_image.rows,
                    static_cast<int>(rgb_image.step),
                    QImage::Format_RGB888);

    label->setPixmap(QPixmap::fromImage(qt_image.copy()));
}

void ImageGUI::saveResult()
{
    if (result_image.empty()) {
        QMessageBox::warning(this, "警告", "没有可保存的结果");
        return;
    }

    QString file_path = QFileDialog::getSaveFileName(
        this,
        "保存文件",
        "",
        "PNG文件 (*.png);;JPG文件 (*.jpg);;所有文件 (*)");

    if (!file_path.isEmpty()) {
        bool success = imwrite(file_path.toStdString(), result_image);
        if (success) {
            QMessageBox::information(this, "成功", "图像已保存到: " + file_path);
        } else {
            QMessageBox::critical(this, "错误", "保存失败");
        }
    }
}
