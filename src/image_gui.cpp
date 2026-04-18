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
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QAbstractItemView>
#include <QPixmap>
#include <QImage>
#include <QApplication>
#include <cstring>
#include <array>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "four_colors.hpp"
#include "segmentation.hpp"

using namespace cv;
using namespace std;

ImageGUI::ImageGUI(QWidget *parent)
    : QMainWindow(parent), original_image(), result_image(), k_value(25), tab_widget(nullptr), info_table(nullptr)
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

    // 中部布局：左侧图像 + 右侧坐标侧栏
    QHBoxLayout *content_layout = new QHBoxLayout();

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

    // Task2 着色结果
    coloring_label = new QLabel();
    coloring_label->setStyleSheet("border: 1px solid #ccc;");
    coloring_label->setMinimumHeight(400);
    coloring_label->setMinimumWidth(600);
    coloring_label->setAlignment(Qt::AlignCenter);
    coloring_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QScrollArea *coloring_scroll = new QScrollArea();
    coloring_scroll->setWidget(coloring_label);
    coloring_scroll->setWidgetResizable(true);
    tab_widget->addTab(coloring_scroll, "着色结果");

    content_layout->addWidget(tab_widget, 1);

    // 右侧栏：种子点编号与坐标表
    QWidget *sidebar = new QWidget();
    sidebar->setMinimumWidth(260);
    sidebar->setMaximumWidth(360);
    QVBoxLayout *sidebar_layout = new QVBoxLayout(sidebar);

        QLabel *sidebar_title = new QLabel("区域信息");
    sidebar_title->setStyleSheet("font-weight: bold; color: #333; padding: 4px 0;");
    sidebar_layout->addWidget(sidebar_title);

        info_table = new QTableWidget();
        info_table->setColumnCount(3);
        info_table->setHorizontalHeaderLabels(QStringList() << "区域" << "颜色" << "邻居数");
        info_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        info_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        info_table->setSelectionMode(QAbstractItemView::SingleSelection);
        info_table->setAlternatingRowColors(true);
        info_table->verticalHeader()->setVisible(false);
        info_table->horizontalHeader()->setStretchLastSection(true);
        info_table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        info_table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        info_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        info_table->setMinimumHeight(420);
        connect(info_table, &QTableWidget::cellClicked,
            this, &ImageGUI::onInfoTableCellClicked);
        sidebar_layout->addWidget(info_table, 1);

    content_layout->addWidget(sidebar);

    main_layout->addLayout(content_layout, 1);

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

    // Task2 按钮
    task2_button = new QPushButton("Task2重着色");
    connect(task2_button, &QPushButton::clicked, this, &ImageGUI::startTask2Recolor);
    task2_button->setStyleSheet(
        "QPushButton {"
        "    background-color: #FF9800;"
        "    color: white;"
        "    border: none;"
        "    padding: 8px 20px;"
        "    border-radius: 4px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #f08a00;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #d97706;"
        "}");
    layout->addWidget(task2_button);

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
            updateSeedTable(vector<Point>());
            updateRegionInfoTable({}, {}, {});
            task2_base_image.release();
            task2_markers.release();
            task2_region_labels.clear();
            task2_node_colors.clear();
            task2_adjacency.clear();
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
    task2_button->setEnabled(false);
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

        status_label->setText("执行分水岭算法");
        qApp->processEvents();

        // 分水岭
        watershed(original_image, markers);

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
        for (size_t i = 0; i < seedPoints.size(); ++i) {
            const auto &pt = seedPoints[i];
            circle(result, pt, 3, Scalar(0, 0, 255), -1);
            circle(result, pt, 3, Scalar(255, 255, 255), 1);

            // 在点旁边标注编号，便于与右侧表格对应
            putText(result,
                    to_string(static_cast<int>(i + 1)),
                    Point(pt.x + 5, pt.y - 5),
                    FONT_HERSHEY_SIMPLEX,
                    0.4,
                    Scalar(255, 255, 255),
                    1,
                    LINE_AA);
        }

        updateSeedTable(seedPoints);

        // 添加时间信息
        string timeText = "Time: " + to_string(duration.count()) + " ms";
        putText(result, timeText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 255, 0), 2);

        result_image = result;
        displayImage(result, result_label);
        tab_widget->setCurrentIndex(1);

        // Task1 不使用区域侧栏信息
        updateRegionInfoTable({}, {}, {});

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
    task2_button->setEnabled(true);
    file_button->setEnabled(true);
    k_spinbox->setEnabled(true);
}

void ImageGUI::startTask2Recolor()
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
    task2_button->setEnabled(false);
    file_button->setEnabled(false);
    k_spinbox->setEnabled(false);

    status_label->setText("Task2: 正在处理");

    try {
        auto start_time = chrono::high_resolution_clock::now();

        status_label->setText("Task2: 生成种子点");
        qApp->processEvents();

        const int width = original_image.cols;
        const int height = original_image.rows;
        vector<Point> seedPoints = SeedSampler::generateSeeds(width, height, k_value);

        if (seedPoints.empty()) {
            throw runtime_error("种子点生成失败");
        }

        Mat markers = Mat::zeros(original_image.size(), CV_32S);
        for (size_t i = 0; i < seedPoints.size(); ++i) {
            circle(markers, seedPoints[i], 1, Scalar(static_cast<int>(i + 1)), -1);
        }

        status_label->setText("Task2: 执行分水岭");
        qApp->processEvents();

        Mat shifted;
        pyrMeanShiftFiltering(original_image, shifted, 10, 51);
        watershed(shifted, markers);

        status_label->setText("Task2: 构建邻接图并四原色着色");
        qApp->processEvents();

        vector<int> regionLabels;
        unordered_map<int, int> labelToNode;
        fourcolor::AdjList adjacency =
            fourcolor::buildAdjacencyFromWatershed(markers, regionLabels, labelToNode);
        const vector<int> bfsOrder = fourcolor::buildBfsOrder(adjacency);

        fourcolor::ColoringOptions options;
        if (k_value >= 1000) {
            options.maxBacktracks = 450000;
            options.maxMs = 3600.0;
        } else if (k_value >= 500) {
            options.maxBacktracks = 250000;
            options.maxMs = 1800.0;
        } else {
            options.maxBacktracks = 120000;
            options.maxMs = 450.0;
        }

        vector<int> nodeColors;
        const fourcolor::ColoringStats cstats =
            fourcolor::colorWithBfsStackBacktracking(adjacency, bfsOrder, nodeColors, options);
        if (!cstats.success) {
            throw runtime_error("四原色回溯着色失败");
        }

        const array<Vec3b, 4> palette = {
            Vec3b(50, 50, 230),
            Vec3b(50, 180, 50),
            Vec3b(230, 80, 40),
            Vec3b(40, 220, 220)
        };

        Mat recolored = Mat::zeros(original_image.size(), CV_8UC3);
        for (int y = 0; y < markers.rows; ++y) {
            for (int x = 0; x < markers.cols; ++x) {
                const int label = markers.at<int>(y, x);
                if (label == -1) {
                    recolored.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
                } else if (label > 0) {
                    auto it = labelToNode.find(label);
                    if (it != labelToNode.end()) {
                        recolored.at<Vec3b>(y, x) = palette[nodeColors[it->second]];
                    }
                }
            }
        }

        Mat result;
        addWeighted(original_image, 0.55, recolored, 0.45, 0.0, result);

        size_t edgeCount = 0;
        int maxDegree = 0;
        for (const auto &nbs : adjacency) {
            edgeCount += nbs.size();
            maxDegree = max(maxDegree, static_cast<int>(nbs.size()));
        }
        edgeCount /= 2;

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end_time - start_time;

        string infoText = "Task2 Time: " + to_string(duration.count()) + " ms";
        putText(result, infoText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 255), 2);

        // Task2 保存基础结果与图结构信息，用于侧栏点击高亮。
        task2_base_image = result;
        task2_markers = markers.clone();
        task2_region_labels = regionLabels;
        task2_node_colors = nodeColors;
        task2_adjacency = adjacency;

        result_image = task2_base_image;
        displayImage(task2_base_image, coloring_label);
        updateRegionInfoTable(task2_region_labels, task2_node_colors, task2_adjacency);
        tab_widget->setCurrentIndex(2);

        status_label->setText(QString("Task2完成! 耗时: %1 ms, 区域: %2, 边: %3, 回溯: %4")
                                  .arg(duration.count(), 0, 'f', 2)
                                  .arg(regionLabels.size())
                                  .arg(edgeCount)
                                  .arg(cstats.backtracks));

        save_button->setEnabled(true);

    } catch (const exception &e) {
        QMessageBox::critical(this, "错误", QString("Task2处理失败: %1").arg(e.what()));
        status_label->setText("Task2处理失败");
    }

    // 启用按钮
    segment_button->setEnabled(true);
    task2_button->setEnabled(true);
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

void ImageGUI::updateSeedTable(const vector<Point> &seedPoints)
{
    // Task1 仍保留点缓存，但侧栏不再显示坐标表。
    current_seed_points = seedPoints;
}

void ImageGUI::updateRegionInfoTable(const vector<int> &regionLabels,
                                     const vector<int> &nodeColors,
                                     const vector<vector<int>> &adjacency)
{
    if (!info_table) {
        return;
    }

    info_table->clearContents();
    info_table->setRowCount(static_cast<int>(regionLabels.size()));

    auto colorName = [](int c) {
        switch (c) {
            case 0: return QString("红");
            case 1: return QString("绿");
            case 2: return QString("蓝");
            case 3: return QString("黄");
            default: return QString("未知");
        }
    };

    auto colorBrush = [](int c) {
        switch (c) {
            case 0: return QBrush(QColor(230, 50, 50));
            case 1: return QBrush(QColor(50, 180, 50));
            case 2: return QBrush(QColor(50, 90, 230));
            case 3: return QBrush(QColor(220, 180, 40));
            default: return QBrush(QColor(180, 180, 180));
        }
    };

    for (int i = 0; i < static_cast<int>(regionLabels.size()); ++i) {
        QTableWidgetItem *region_item = new QTableWidgetItem(QString::number(regionLabels[i]));
        region_item->setTextAlignment(Qt::AlignCenter);
        info_table->setItem(i, 0, region_item);

        const int c = (i < static_cast<int>(nodeColors.size())) ? nodeColors[i] : -1;
        QTableWidgetItem *color_item = new QTableWidgetItem(colorName(c));
        color_item->setTextAlignment(Qt::AlignCenter);
        color_item->setBackground(colorBrush(c));
        color_item->setForeground(QBrush(Qt::white));
        info_table->setItem(i, 1, color_item);

        const int degree = (i < static_cast<int>(adjacency.size())) ? static_cast<int>(adjacency[i].size()) : 0;
        QTableWidgetItem *neighbor_item = new QTableWidgetItem(QString::number(degree));
        neighbor_item->setTextAlignment(Qt::AlignCenter);
        info_table->setItem(i, 2, neighbor_item);
    }

    info_table->clearSelection();
}

Mat ImageGUI::brightenRegionByLabel(const Mat &baseImage,
                                    const Mat &markers,
                                    int targetLabel)
{
    Mat highlighted = baseImage.clone();
    if (highlighted.empty() || markers.empty() || targetLabel <= 0) {
        return highlighted;
    }

    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            if (markers.at<int>(y, x) == targetLabel) {
                Vec3b &px = highlighted.at<Vec3b>(y, x);
                px[0] = saturate_cast<uchar>(px[0] * 1.35);
                px[1] = saturate_cast<uchar>(px[1] * 1.35);
                px[2] = saturate_cast<uchar>(px[2] * 1.35);
            }
        }
    }

    return highlighted;
}

void ImageGUI::onInfoTableCellClicked(int row, int column)
{
    (void)column;

    if (task2_base_image.empty() || task2_markers.empty()) {
        return;
    }
    if (row < 0 || row >= static_cast<int>(task2_region_labels.size())) {
        return;
    }

    const int label = task2_region_labels[row];
    const int colorIdx = (row < static_cast<int>(task2_node_colors.size())) ? task2_node_colors[row] : -1;
    const int neighborCount = (row < static_cast<int>(task2_adjacency.size()))
                                  ? static_cast<int>(task2_adjacency[row].size())
                                  : 0;

    Mat highlighted = brightenRegionByLabel(task2_base_image, task2_markers, label);

    const char *colorName = "Unknown";
    if (colorIdx == 0) colorName = "Red";
    else if (colorIdx == 1) colorName = "Green";
    else if (colorIdx == 2) colorName = "Blue";
    else if (colorIdx == 3) colorName = "Yellow";

    const string info = "Region " + to_string(label) + "  Color: " + colorName +
                        "  Neighbors: " + to_string(neighborCount);
    putText(highlighted, info, Point(20, 75), FONT_HERSHEY_SIMPLEX,
            0.6, Scalar(0, 255, 255), 2, LINE_AA);

    displayImage(highlighted, coloring_label);
    tab_widget->setCurrentIndex(2);
    status_label->setText(QString::fromStdString("Selected " + info));
}
