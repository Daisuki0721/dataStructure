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
#include <QTabBar>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QAbstractItemView>
#include <QStackedWidget>
#include <QPixmap>
#include <QImage>
#include <QApplication>
#include <QEvent>
#include <QMouseEvent>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <cstring>
#include <array>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "four_colors.hpp"
#include "segmentation.hpp"

using namespace cv;
using namespace std;

namespace {

struct GuiRegionArea {
    int label = 0;
    int area = 0;
};

struct GuiHuffmanNode {
    int weight = 0;
    int label = -1;
    GuiHuffmanNode *left = nullptr;
    GuiHuffmanNode *right = nullptr;
};

vector<GuiRegionArea> buildRegionAreasFromMarkers(const Mat &markers)
{
    unordered_map<int, int> areaByLabel;
    for (int y = 0; y < markers.rows; ++y) {
        const int *row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            const int label = row[x];
            if (label > 0) {
                areaByLabel[label]++;
            }
        }
    }

    vector<GuiRegionArea> areas;
    areas.reserve(areaByLabel.size());
    for (const auto &kv : areaByLabel) {
        areas.push_back(GuiRegionArea{kv.first, kv.second});
    }

    sort(areas.begin(), areas.end(), [](const GuiRegionArea &a, const GuiRegionArea &b) {
        if (a.area != b.area) {
            return a.area < b.area;
        }
        return a.label < b.label;
    });

    return areas;
}

pair<int, int> findAreaRange(const vector<GuiRegionArea> &sortedAreas, int low, int high)
{
    if (sortedAreas.empty() || low > high) {
        return {-1, -1};
    }

    auto lb = lower_bound(sortedAreas.begin(), sortedAreas.end(), low,
                          [](const GuiRegionArea &r, int value) { return r.area < value; });
    auto ub = upper_bound(sortedAreas.begin(), sortedAreas.end(), high,
                          [](int value, const GuiRegionArea &r) { return value < r.area; });

    if (lb == sortedAreas.end() || lb >= ub) {
        return {-1, -1};
    }

    const int first = static_cast<int>(distance(sortedAreas.begin(), lb));
    const int last = static_cast<int>(distance(sortedAreas.begin(), ub)) - 1;
    return {first, last};
}

unordered_map<int, Point> computeRegionCentroids(const Mat &markers,
                                                 const vector<GuiRegionArea> &selected)
{
    set<int> selectedLabels;
    for (const auto &r : selected) {
        selectedLabels.insert(r.label);
    }

    struct Acc {
        long long sx = 0;
        long long sy = 0;
        int count = 0;
    };

    unordered_map<int, Acc> acc;
    for (int y = 0; y < markers.rows; ++y) {
        const int *row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            const int label = row[x];
            if (selectedLabels.find(label) != selectedLabels.end()) {
                auto &a = acc[label];
                a.sx += x;
                a.sy += y;
                a.count++;
            }
        }
    }

    unordered_map<int, Point> centroids;
    for (const auto &kv : acc) {
        if (kv.second.count > 0) {
            centroids[kv.first] = Point(static_cast<int>(kv.second.sx / kv.second.count),
                                        static_cast<int>(kv.second.sy / kv.second.count));
        }
    }
    return centroids;
}

void putCenteredText(Mat &image,
                     const string &text,
                     const Point &center,
                     double fontScale,
                     const Scalar &color,
                     int thickness)
{
    int baseline = 0;
    const Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    Point org(center.x - textSize.width / 2, center.y + textSize.height / 2);

    org.x = max(0, min(org.x, image.cols - textSize.width - 1));
    org.y = max(textSize.height, min(org.y, image.rows - 1));

    putText(image, text, org, FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, LINE_AA);
}

struct QueueNode {
    GuiHuffmanNode *node = nullptr;
    int minLeafLabel = -1;
    int serial = 0;
};

struct QueueNodeCmp {
    bool operator()(const QueueNode &a, const QueueNode &b) const
    {
        if (a.node->weight != b.node->weight) {
            return a.node->weight > b.node->weight;
        }
        if (a.minLeafLabel != b.minLeafLabel) {
            return a.minLeafLabel > b.minLeafLabel;
        }
        return a.serial > b.serial;
    }
};

GuiHuffmanNode *buildHuffmanTree(const vector<GuiRegionArea> &selected)
{
    if (selected.empty()) {
        return nullptr;
    }

    priority_queue<QueueNode, vector<QueueNode>, QueueNodeCmp> pq;
    int serial = 0;
    for (const auto &r : selected) {
        auto *leaf = new GuiHuffmanNode();
        leaf->weight = r.area;
        leaf->label = r.label;
        pq.push(QueueNode{leaf, r.label, serial++});
    }

    if (pq.size() == 1) {
        return pq.top().node;
    }

    while (pq.size() > 1) {
        QueueNode left = pq.top();
        pq.pop();
        QueueNode right = pq.top();
        pq.pop();

        auto *parent = new GuiHuffmanNode();
        parent->weight = left.node->weight + right.node->weight;
        parent->label = -1;
        parent->left = left.node;
        parent->right = right.node;

        const int minLeafLabel = min(left.minLeafLabel, right.minLeafLabel);
        pq.push(QueueNode{parent, minLeafLabel, serial++});
    }

    return pq.top().node;
}

void generateHuffmanCodes(const GuiHuffmanNode *root,
                          const string &prefix,
                          map<int, string> &outCodes,
                          int depth,
                          int &maxDepth)
{
    if (!root) {
        return;
    }

    maxDepth = max(maxDepth, depth);
    if (!root->left && !root->right) {
        outCodes[root->label] = prefix.empty() ? "0" : prefix;
        return;
    }

    generateHuffmanCodes(root->left, prefix + "0", outCodes, depth + 1, maxDepth);
    generateHuffmanCodes(root->right, prefix + "1", outCodes, depth + 1, maxDepth);
}

int countHuffmanLeaves(const GuiHuffmanNode *root)
{
    if (!root) {
        return 0;
    }
    if (!root->left && !root->right) {
        return 1;
    }
    return countHuffmanLeaves(root->left) + countHuffmanLeaves(root->right);
}

void drawHuffmanTree(const GuiHuffmanNode *root,
                     Mat &canvas,
                     int x,
                     int y,
                     int xOffset,
                     int levelHeight,
                     vector<Point> *nodeCenters)
{
    if (!root) {
        return;
    }

    const Scalar edgeColor(90, 90, 90);
    const Scalar nodeColor(255, 255, 255);
    const Scalar textColor(30, 30, 30);

    if (root->left) {
        const int childX = x - xOffset;
        const int childY = y + levelHeight;
        line(canvas, Point(x, y), Point(childX, childY), edgeColor, 1, LINE_AA);
        putText(canvas, "0", Point((x + childX) / 2 - 8, (y + childY) / 2),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(120, 120, 120), 1, LINE_AA);
        drawHuffmanTree(root->left,
                        canvas,
                        childX,
                        childY,
                        max(24, xOffset / 2),
                        levelHeight,
                        nodeCenters);
    }
    if (root->right) {
        const int childX = x + xOffset;
        const int childY = y + levelHeight;
        line(canvas, Point(x, y), Point(childX, childY), edgeColor, 1, LINE_AA);
        putText(canvas, "1", Point((x + childX) / 2 + 2, (y + childY) / 2),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(120, 120, 120), 1, LINE_AA);
        drawHuffmanTree(root->right,
                        canvas,
                        childX,
                        childY,
                        max(24, xOffset / 2),
                        levelHeight,
                        nodeCenters);
    }

    if (nodeCenters) {
        nodeCenters->push_back(Point(x, y));
    }

    circle(canvas, Point(x, y), 16, nodeColor, FILLED, LINE_AA);
    circle(canvas, Point(x, y), 16, Scalar(120, 120, 120), 1, LINE_AA);

    if (!root->left && !root->right) {
        const string text = to_string(root->weight);
        putCenteredText(canvas, text, Point(x, y), 0.35, textColor, 1);
    }
}

void destroyHuffmanTree(GuiHuffmanNode *root)
{
    if (!root) {
        return;
    }
    destroyHuffmanTree(root->left);
    destroyHuffmanTree(root->right);
    delete root;
}

}  // namespace

ImageGUI::ImageGUI(QWidget *parent)
    : QMainWindow(parent), original_image(), result_image(), k_value(25), tab_widget(nullptr),
      sidebar_widget(nullptr), sidebar_stack(nullptr), seed_table(nullptr), info_table(nullptr),
      huffman_table(nullptr), huffman_code_table(nullptr), huffman_view_index(0),
      huffman_selected_area_label(-1), huffman_selected_tree_node_index(-1),
      huffman_last_low(25), huffman_last_high(1000),
      original_label(nullptr), result_label(nullptr), coloring_label(nullptr), huffman_label(nullptr),
      huffman_view_title_label(nullptr), sidebar_title_label(nullptr), file_label(nullptr),
      status_label(nullptr), file_button(nullptr), segment_button(nullptr), task2_button(nullptr),
      task3_button(nullptr), huffman_area_button(nullptr), huffman_tree_button(nullptr),
      save_button(nullptr), k_spinbox(nullptr), view_mode_combo(nullptr)
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

    // Task3 Huffman结果页：顶部按钮切换“面积区域”与“Huffman树”。
    QWidget *huffman_page = new QWidget();
    QVBoxLayout *huffman_layout = new QVBoxLayout(huffman_page);
    QHBoxLayout *huffman_switch_layout = new QHBoxLayout();

    huffman_area_button = new QPushButton("面积区域");
    huffman_area_button->setCheckable(true);
    connect(huffman_area_button, &QPushButton::clicked, this, &ImageGUI::onHuffmanAreaViewClicked);

    huffman_view_title_label = new QLabel("高亮区域");
    huffman_view_title_label->setAlignment(Qt::AlignCenter);
    huffman_view_title_label->setStyleSheet("font-weight: bold; color: #333;");

    huffman_tree_button = new QPushButton("Huffman树");
    huffman_tree_button->setCheckable(true);
    connect(huffman_tree_button, &QPushButton::clicked, this, &ImageGUI::onHuffmanTreeViewClicked);

    huffman_switch_layout->addWidget(huffman_area_button);
    huffman_switch_layout->addWidget(huffman_view_title_label, 1);
    huffman_switch_layout->addWidget(huffman_tree_button);
    huffman_layout->addLayout(huffman_switch_layout);

    huffman_label = new QLabel();
    huffman_label->setStyleSheet("border: 1px solid #ccc;");
    huffman_label->setMinimumHeight(400);
    huffman_label->setMinimumWidth(600);
    huffman_label->setAlignment(Qt::AlignCenter);
    huffman_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QScrollArea *huffman_scroll = new QScrollArea();
    huffman_scroll->setWidget(huffman_label);
    huffman_scroll->setWidgetResizable(true);
    huffman_layout->addWidget(huffman_scroll, 1);
    huffman_label->installEventFilter(this);

    tab_widget->addTab(huffman_page, "Huffman排序");

    // 隐藏原有页签栏，改用固定下拉框切换观察模式。
    tab_widget->tabBar()->hide();
    connect(tab_widget, &QTabWidget::currentChanged, this, &ImageGUI::onTabChanged);

    content_layout->addWidget(tab_widget, 1);

    // 右侧栏：按页签切换“采样点坐标”与“着色信息”
    sidebar_widget = new QWidget();
    sidebar_widget->setMinimumWidth(260);
    sidebar_widget->setMaximumWidth(360);
    QVBoxLayout *sidebar_layout = new QVBoxLayout(sidebar_widget);

    sidebar_title_label = new QLabel("采样点坐标");
    sidebar_title_label->setStyleSheet("font-weight: bold; color: #333; padding: 4px 0;");
    sidebar_layout->addWidget(sidebar_title_label);

    sidebar_stack = new QStackedWidget();

    seed_table = new QTableWidget();
    seed_table->setColumnCount(2);
    seed_table->setHorizontalHeaderLabels(QStringList() << "编号" << "坐标");
    seed_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    seed_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    seed_table->setSelectionMode(QAbstractItemView::SingleSelection);
    seed_table->setAlternatingRowColors(true);
    seed_table->verticalHeader()->setVisible(false);
    seed_table->horizontalHeader()->setStretchLastSection(true);
    seed_table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    seed_table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    seed_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    seed_table->setMinimumHeight(420);
        connect(seed_table, &QTableWidget::cellClicked,
            this, &ImageGUI::onSeedTableCellClicked);
    sidebar_stack->addWidget(seed_table);

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
    sidebar_stack->addWidget(info_table);

        huffman_table = new QTableWidget();
        huffman_table->setColumnCount(2);
        huffman_table->setHorizontalHeaderLabels(QStringList() << "编号" << "面积");
        huffman_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        huffman_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        huffman_table->setSelectionMode(QAbstractItemView::SingleSelection);
        huffman_table->setAlternatingRowColors(true);
        huffman_table->verticalHeader()->setVisible(false);
        huffman_table->horizontalHeader()->setStretchLastSection(true);
        huffman_table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        huffman_table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        huffman_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        huffman_table->setMinimumHeight(420);
            connect(huffman_table, &QTableWidget::cellClicked,
                this, &ImageGUI::onHuffmanAreaTableCellClicked);
        sidebar_stack->addWidget(huffman_table);

        huffman_code_table = new QTableWidget();
        huffman_code_table->setColumnCount(3);
        huffman_code_table->setHorizontalHeaderLabels(QStringList() << "编号" << "面积" << "编码");
        huffman_code_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        huffman_code_table->setSelectionBehavior(QAbstractItemView::SelectRows);
        huffman_code_table->setSelectionMode(QAbstractItemView::SingleSelection);
        huffman_code_table->setAlternatingRowColors(true);
        huffman_code_table->verticalHeader()->setVisible(false);
        huffman_code_table->horizontalHeader()->setStretchLastSection(true);
        huffman_code_table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
        huffman_code_table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        huffman_code_table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        huffman_code_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        huffman_code_table->setMinimumHeight(420);
        sidebar_stack->addWidget(huffman_code_table);

    sidebar_layout->addWidget(sidebar_stack, 1);

    content_layout->addWidget(sidebar_widget);

    main_layout->addLayout(content_layout, 1);

    // 状态信息
    status_label = new QLabel("准备就绪");
    status_label->setStyleSheet("color: #666; padding: 10px;");
    main_layout->addWidget(status_label);

    show();

    // 初始为原图页：隐藏侧栏。
    syncSidebarByTab(0);
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

        // 观察模式下拉菜单（固定位置）
        layout->addWidget(new QLabel("观察模式:"));
        view_mode_combo = new QComboBox();
        view_mode_combo->addItem("原始图像");
        view_mode_combo->addItem("分割结果");
        view_mode_combo->addItem("着色结果");
        view_mode_combo->addItem("Huffman排序");
        view_mode_combo->setCurrentIndex(0);
        view_mode_combo->setMinimumWidth(180);
        connect(view_mode_combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ImageGUI::onViewModeChanged);
        layout->addWidget(view_mode_combo);

        layout->addSpacing(20);

    // K值输入
    layout->addWidget(new QLabel("K值 (种子数):"));
    k_spinbox = new QSpinBox();
    k_spinbox->setMinimum(25);
    k_spinbox->setMaximum(1000);
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
    task2_button = new QPushButton("重着色");
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

    // Task3 按钮
    task3_button = new QPushButton("Huffman排序");
    connect(task3_button, &QPushButton::clicked, this, &ImageGUI::startTask3HuffmanSort);
    task3_button->setStyleSheet(
        "QPushButton {"
        "    background-color: #009688;"
        "    color: white;"
        "    border: none;"
        "    padding: 8px 20px;"
        "    border-radius: 4px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #00897b;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #00796b;"
        "}");
    layout->addWidget(task3_button);

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
            updateHuffmanTable({}, {});
            updateHuffmanCodeTable({});
            task1_cached_image.release();
            task1_display_image.release();
            task2_display_image.release();
            task2_base_image.release();
            task2_markers.release();
            task2_region_labels.clear();
            task2_node_colors.clear();
            task2_adjacency.clear();
            huffman_highlight_image.release();
            huffman_tree_image.release();
            huffman_markers.release();
            huffman_view_index = 0;
            huffman_selected_area_label = -1;
            huffman_selected_tree_node_index = -1;
            huffman_tree_node_centers.clear();
            huffman_all_label_areas.clear();
            huffman_matched_labels.clear();
            huffman_code_rows.clear();
            refreshHuffmanView();
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
    task3_button->setEnabled(false);
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
        task1_cached_image = result.clone();
        task1_display_image = task1_cached_image.clone();
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
    task3_button->setEnabled(true);
    file_button->setEnabled(true);
    k_spinbox->setEnabled(true);
}

void ImageGUI::onSeedTableCellClicked(int row, int column)
{
    (void)column;

    if (task1_cached_image.empty()) {
        return;
    }
    if (row < 0 || row >= static_cast<int>(current_seed_points.size())) {
        return;
    }

    Mat highlighted = task1_cached_image.clone();
    const Point &pt = current_seed_points[row];

    circle(highlighted, pt, 10, Scalar(0, 255, 255), 2);
    circle(highlighted, pt, 4, Scalar(0, 255, 255), -1);

    const string info = "Point #" + to_string(row + 1) +
                        " (" + to_string(pt.x) + "," + to_string(pt.y) + ")";
    putText(highlighted, info, Point(20, 75), FONT_HERSHEY_SIMPLEX,
            0.6, Scalar(0, 255, 255), 2, LINE_AA);

    task1_display_image = highlighted;
    result_image = task1_display_image;
    displayImage(task1_display_image, result_label);
    tab_widget->setCurrentIndex(1);
    status_label->setText(QString::fromStdString("已选中 " + info));
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
    task3_button->setEnabled(false);
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

        // 失败时快速重试：每轮最多 250ms，总预算 5s。
        const double totalRetryBudgetMs = 5000.0;
        const double singleAttemptMs = 250.0;
        auto retryStart = chrono::steady_clock::now();
        mt19937 rng(static_cast<unsigned int>(chrono::high_resolution_clock::now().time_since_epoch().count()));

        fourcolor::ColoringStats cstats;
        bool colored = false;
        int attempt = 0;
        while (true) {
            const double spentMs = chrono::duration<double, milli>(chrono::steady_clock::now() - retryStart).count();
            if (spentMs > totalRetryBudgetMs) {
                break;
            }

            vector<int> attemptOrder = bfsOrder;
            if (attempt > 0 && !attemptOrder.empty()) {
                shuffle(attemptOrder.begin(), attemptOrder.end(), rng);
            }

            fourcolor::ColoringOptions attemptOptions = options;
            attemptOptions.maxMs = min(options.maxMs, singleAttemptMs);

            cstats = fourcolor::colorWithBfsStackBacktracking(
                adjacency, attemptOrder, nodeColors, attemptOptions);

            if (cstats.success) {
                colored = true;
                break;
            }
            ++attempt;
            status_label->setText(QString("Task2: 着色重试中 (%1)").arg(attempt));
            qApp->processEvents();
        }
        if (!colored) {
            throw runtime_error("四原色回溯着色失败（5秒内重试仍未成功）");
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

        string infoText = "Time: " + to_string(duration.count()) + " ms";
        putText(result, infoText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 255), 2);

        // Task2 保存基础结果与图结构信息，用于侧栏点击高亮。
        task2_base_image = result;
        task2_display_image = task2_base_image.clone();
        task2_markers = markers.clone();
        task2_region_labels = regionLabels;
        task2_node_colors = nodeColors;
        task2_adjacency = adjacency;

        result_image = task2_display_image;
        displayImage(task2_display_image, coloring_label);
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
    task3_button->setEnabled(true);
    file_button->setEnabled(true);
    k_spinbox->setEnabled(true);
}

void ImageGUI::startTask3HuffmanSort()
{
    if (image_path.empty()) {
        QMessageBox::warning(this, "警告", "请先选择图像文件");
        return;
    }

    if (original_image.empty()) {
        QMessageBox::warning(this, "警告", "图像为空");
        return;
    }

    segment_button->setEnabled(false);
    task2_button->setEnabled(false);
    task3_button->setEnabled(false);
    file_button->setEnabled(false);
    k_spinbox->setEnabled(false);

    GuiHuffmanNode *root = nullptr;

    try {
        auto start_time = chrono::high_resolution_clock::now();
        status_label->setText("Task3: 生成分水岭区域");
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

        Mat shifted;
        pyrMeanShiftFiltering(original_image, shifted, 10, 51);
        watershed(shifted, markers);

        vector<GuiRegionArea> allAreas = buildRegionAreasFromMarkers(markers);
        if (allAreas.empty()) {
            throw runtime_error("分水岭未得到有效区域");
        }

        const int minArea = allAreas.front().area;
        const int maxArea = allAreas.back().area;

        QDialog rangeDialog(this);
        rangeDialog.setWindowTitle("Huffman排序 - 面积范围");
        QFormLayout form(&rangeDialog);

        QSpinBox minSpin;
        minSpin.setRange(minArea, maxArea);
        minSpin.setValue(max(minArea, min(maxArea, huffman_last_low)));
        form.addRow("最小面积:", &minSpin);

        QSpinBox maxSpin;
        maxSpin.setRange(minArea, maxArea);
        maxSpin.setValue(max(minArea, min(maxArea, huffman_last_high)));
        form.addRow("最大面积:", &maxSpin);

        QDialogButtonBox buttons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                                 Qt::Horizontal,
                                 &rangeDialog);
        form.addWidget(&buttons);
        connect(&buttons, &QDialogButtonBox::accepted, &rangeDialog, &QDialog::accept);
        connect(&buttons, &QDialogButtonBox::rejected, &rangeDialog, &QDialog::reject);

        if (rangeDialog.exec() != QDialog::Accepted) {
            status_label->setText("Task3: 已取消");
            segment_button->setEnabled(true);
            task2_button->setEnabled(true);
            task3_button->setEnabled(true);
            file_button->setEnabled(true);
            k_spinbox->setEnabled(true);
            return;
        }

        int low = minSpin.value();
        int high = maxSpin.value();
        if (low > high) {
            swap(low, high);
        }
        huffman_last_low = low;
        huffman_last_high = high;

        const pair<int, int> range = findAreaRange(allAreas, low, high);
        if (range.first < 0) {
            QMessageBox::information(this, "提示", "当前面积范围内没有匹配区域");
            status_label->setText("Task3: 未找到匹配区域");
            segment_button->setEnabled(true);
            task2_button->setEnabled(true);
            task3_button->setEnabled(true);
            file_button->setEnabled(true);
            k_spinbox->setEnabled(true);
            return;
        }

        vector<GuiRegionArea> selected;
        selected.reserve(static_cast<size_t>(range.second - range.first + 1));
        for (int i = range.first; i <= range.second; ++i) {
            selected.push_back(allAreas[i]);
        }

        huffman_selected_area_label = -1;
        huffman_selected_tree_node_index = -1;
        huffman_markers = markers.clone();

        huffman_matched_labels.clear();
        for (const auto &r : selected) {
            huffman_matched_labels.insert(r.label);
        }

        unordered_map<int, Vec3b> colorByLabel;
        mt19937 rng(static_cast<unsigned int>(chrono::high_resolution_clock::now().time_since_epoch().count()));
        uniform_int_distribution<int> ch(80, 255);
        for (const auto &r : selected) {
            colorByLabel[r.label] = Vec3b(static_cast<uchar>(ch(rng)),
                                          static_cast<uchar>(ch(rng)),
                                          static_cast<uchar>(ch(rng)));
        }

        Mat highlightMask = Mat::zeros(original_image.size(), CV_8UC3);
        for (int y = 0; y < markers.rows; ++y) {
            for (int x = 0; x < markers.cols; ++x) {
                const int label = markers.at<int>(y, x);
                if (label == -1) {
                    highlightMask.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
                } else if (huffman_matched_labels.find(label) != huffman_matched_labels.end()) {
                    highlightMask.at<Vec3b>(y, x) = colorByLabel[label];
                }
            }
        }

        Mat highlighted;
        addWeighted(original_image, 0.45, highlightMask, 0.55, 0.0, highlighted);

        const auto centroids = computeRegionCentroids(markers, selected);
        for (const auto &r : selected) {
            auto it = centroids.find(r.label);
            if (it != centroids.end()) {
                putCenteredText(highlighted,
                                to_string(r.area),
                                it->second,
                                0.48,
                                Scalar(255, 255, 255),
                                1);
            }
        }

        root = buildHuffmanTree(selected);
        if (!root) {
            throw runtime_error("Huffman树构建失败");
        }

        map<int, string> codes;
        int maxDepth = 0;
        generateHuffmanCodes(root, "", codes, 0, maxDepth);

        huffman_code_rows.clear();
        huffman_code_rows.reserve(selected.size());
        for (const auto &r : selected) {
            const auto it = codes.find(r.label);
            if (it != codes.end()) {
                huffman_code_rows.push_back(make_tuple(r.label, r.area, it->second));
            }
        }
        sort(huffman_code_rows.begin(), huffman_code_rows.end(),
             [](const tuple<int, int, string> &a, const tuple<int, int, string> &b) {
                 const string &codeA = get<2>(a);
                 const string &codeB = get<2>(b);
                 if (codeA.size() != codeB.size()) {
                     return codeA.size() < codeB.size();
                 }
                 if (get<1>(a) != get<1>(b)) {
                     return get<1>(a) < get<1>(b);
                 }
                 return get<0>(a) < get<0>(b);
             });

        const int leaves = max(2, countHuffmanLeaves(root));
        const int widthCanvas = max(1200, leaves * 80);
        const int heightCanvas = max(600, (maxDepth + 2) * 120);
        Mat treeCanvas(heightCanvas, widthCanvas, CV_8UC3, Scalar(245, 245, 245));
        huffman_tree_node_centers.clear();
        drawHuffmanTree(root,
                treeCanvas,
                widthCanvas / 2,
                60,
                widthCanvas / 4,
                100,
                &huffman_tree_node_centers);
        putText(treeCanvas,
                "Huffman Tree",
                Point(20, 34),
                FONT_HERSHEY_SIMPLEX,
                0.85,
                Scalar(50, 50, 50),
                2,
                LINE_AA);

        huffman_highlight_image = highlighted;
        huffman_tree_image = treeCanvas;
        huffman_view_index = 0;

        huffman_all_label_areas.clear();
        huffman_all_label_areas.reserve(allAreas.size());
        for (const auto &r : allAreas) {
            huffman_all_label_areas.push_back({r.label, r.area});
        }
        updateHuffmanTable(huffman_all_label_areas, huffman_matched_labels);
        updateHuffmanCodeTable(huffman_code_rows);

        refreshHuffmanView();
        tab_widget->setCurrentIndex(3);
        save_button->setEnabled(true);

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end_time - start_time;
        status_label->setText(QString("Task3完成! 耗时: %1 ms, 匹配区域: %2, 编码节点: %3")
                                  .arg(duration.count(), 0, 'f', 2)
                                  .arg(selected.size())
                                  .arg(codes.size()));

        destroyHuffmanTree(root);
        root = nullptr;
    } catch (const exception &e) {
        if (root) {
            destroyHuffmanTree(root);
            root = nullptr;
        }
        QMessageBox::critical(this, "错误", QString("Task3处理失败: %1").arg(e.what()));
        status_label->setText("Task3处理失败");
    }

    segment_button->setEnabled(true);
    task2_button->setEnabled(true);
    task3_button->setEnabled(true);
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
    current_seed_points = seedPoints;

    if (!seed_table) {
        return;
    }

    seed_table->clearContents();
    seed_table->setRowCount(static_cast<int>(seedPoints.size()));
    for (int i = 0; i < static_cast<int>(seedPoints.size()); ++i) {
        QTableWidgetItem *id_item = new QTableWidgetItem(QString::number(i + 1));
        id_item->setTextAlignment(Qt::AlignCenter);
        seed_table->setItem(i, 0, id_item);

        const Point &p = seedPoints[i];
        QTableWidgetItem *coord_item = new QTableWidgetItem(QString("(%1,%2)").arg(p.x).arg(p.y));
        coord_item->setTextAlignment(Qt::AlignCenter);
        seed_table->setItem(i, 1, coord_item);
    }

    seed_table->clearSelection();
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

void ImageGUI::updateHuffmanTable(const vector<pair<int, int>> &allLabelAreas,
                                  const set<int> &matchedLabels)
{
    if (!huffman_table) {
        return;
    }

    huffman_table->clearContents();
    huffman_table->setRowCount(static_cast<int>(allLabelAreas.size()));

    const QColor hitBg(255, 252, 232);
    const QColor hitText(80, 50, 0);

    for (int i = 0; i < static_cast<int>(allLabelAreas.size()); ++i) {
        const int label = allLabelAreas[i].first;
        const int area = allLabelAreas[i].second;

        QTableWidgetItem *labelItem = new QTableWidgetItem(QString::number(label));
        labelItem->setTextAlignment(Qt::AlignCenter);
        huffman_table->setItem(i, 0, labelItem);

        QTableWidgetItem *areaItem = new QTableWidgetItem(QString::number(area));
        areaItem->setTextAlignment(Qt::AlignCenter);
        huffman_table->setItem(i, 1, areaItem);

        if (matchedLabels.find(label) != matchedLabels.end()) {
            labelItem->setBackground(QBrush(hitBg));
            areaItem->setBackground(QBrush(hitBg));
            labelItem->setForeground(QBrush(hitText));
            areaItem->setForeground(QBrush(hitText));

            QFont boldFont = labelItem->font();
            boldFont.setBold(true);
            labelItem->setFont(boldFont);
            areaItem->setFont(boldFont);
        }
    }

    huffman_table->clearSelection();
}

void ImageGUI::updateHuffmanCodeTable(const vector<tuple<int, int, string>> &codeRows)
{
    if (!huffman_code_table) {
        return;
    }

    huffman_code_table->clearContents();
    huffman_code_table->setRowCount(static_cast<int>(codeRows.size()));

    for (int i = 0; i < static_cast<int>(codeRows.size()); ++i) {
        const int label = get<0>(codeRows[i]);
        const int area = get<1>(codeRows[i]);
        const string &code = get<2>(codeRows[i]);

        QTableWidgetItem *labelItem = new QTableWidgetItem(QString::number(label));
        labelItem->setTextAlignment(Qt::AlignCenter);
        huffman_code_table->setItem(i, 0, labelItem);

        QTableWidgetItem *areaItem = new QTableWidgetItem(QString::number(area));
        areaItem->setTextAlignment(Qt::AlignCenter);
        huffman_code_table->setItem(i, 1, areaItem);

        QTableWidgetItem *codeItem = new QTableWidgetItem(QString::fromStdString(code));
        codeItem->setTextAlignment(Qt::AlignCenter);
        huffman_code_table->setItem(i, 2, codeItem);
    }

    huffman_code_table->clearSelection();
}

void ImageGUI::refreshHuffmanView()
{
    if (!huffman_area_button || !huffman_tree_button || !huffman_view_title_label || !huffman_label) {
        return;
    }

    const bool hasImages = !huffman_highlight_image.empty() && !huffman_tree_image.empty();
    huffman_area_button->setEnabled(hasImages);
    huffman_tree_button->setEnabled(hasImages);

    auto applyButtonStyle = [](QPushButton *btn, bool active) {
        if (!btn) {
            return;
        }
        if (active) {
            btn->setStyleSheet(
                "QPushButton {"
                "  background-color: #1565c0;"
                "  color: white;"
                "  border: none;"
                "  padding: 6px 14px;"
                "  border-radius: 4px;"
                "  font-weight: bold;"
                "}");
        } else {
            btn->setStyleSheet(
                "QPushButton {"
                "  background-color: #eceff1;"
                "  color: #37474f;"
                "  border: 1px solid #cfd8dc;"
                "  padding: 6px 14px;"
                "  border-radius: 4px;"
                "}");
        }
    };

    if (!hasImages) {
        huffman_view_index = 0;
        huffman_area_button->setChecked(true);
        huffman_tree_button->setChecked(false);
        applyButtonStyle(huffman_area_button, true);
        applyButtonStyle(huffman_tree_button, false);
        huffman_view_title_label->setText("高亮区域");
        huffman_label->clear();
        syncSidebarByTab(3);
        return;
    }

    huffman_view_index = (huffman_view_index == 1) ? 1 : 0;
    huffman_area_button->setChecked(huffman_view_index == 0);
    huffman_tree_button->setChecked(huffman_view_index == 1);
    applyButtonStyle(huffman_area_button, huffman_view_index == 0);
    applyButtonStyle(huffman_tree_button, huffman_view_index == 1);

    if (huffman_view_index == 0) {
        Mat shown = huffman_highlight_image.clone();
        if (huffman_selected_area_label > 0 && !huffman_markers.empty()) {
            shown = brightenRegionByLabel(shown, huffman_markers, huffman_selected_area_label);
        }
        huffman_view_title_label->setText("高亮区域");
        displayImage(shown, huffman_label);
        result_image = shown;
    } else {
        Mat shown = huffman_tree_image.clone();
        if (huffman_selected_tree_node_index >= 0 &&
            huffman_selected_tree_node_index < static_cast<int>(huffman_tree_node_centers.size())) {
            const Point center = huffman_tree_node_centers[huffman_selected_tree_node_index];
            circle(shown, center, 24, Scalar(60, 200, 255), 2, LINE_AA);
            circle(shown, center, 17, Scalar(60, 200, 255), 2, LINE_AA);
        }
        huffman_view_title_label->setText("Huffman树");
        displayImage(shown, huffman_label);
        result_image = shown;
    }

    syncSidebarByTab(3);
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

    task2_display_image = highlighted;
    result_image = task2_display_image;
    displayImage(highlighted, coloring_label);
    tab_widget->setCurrentIndex(2);
    status_label->setText(QString::fromStdString("Selected " + info));
}

void ImageGUI::onTabChanged(int index)
{
    if (view_mode_combo && view_mode_combo->currentIndex() != index) {
        view_mode_combo->blockSignals(true);
        view_mode_combo->setCurrentIndex(index);
        view_mode_combo->blockSignals(false);
    }

    syncSidebarByTab(index);

    if (index == 1) {
        if (!task1_display_image.empty()) {
            displayImage(task1_display_image, result_label);
            result_image = task1_display_image;
        } else if (!task1_cached_image.empty()) {
            displayImage(task1_cached_image, result_label);
            result_image = task1_cached_image;
        }
    } else if (index == 2) {
        if (!task2_display_image.empty()) {
            displayImage(task2_display_image, coloring_label);
            result_image = task2_display_image;
        } else if (!task2_base_image.empty()) {
            displayImage(task2_base_image, coloring_label);
            result_image = task2_base_image;
        }
    } else if (index == 3) {
        refreshHuffmanView();
    }
}

void ImageGUI::onViewModeChanged(int index)
{
    if (!tab_widget) {
        return;
    }
    if (index < 0 || index >= tab_widget->count()) {
        return;
    }
    tab_widget->setCurrentIndex(index);
}

void ImageGUI::onHuffmanAreaViewClicked()
{
    if (huffman_highlight_image.empty() || huffman_tree_image.empty()) {
        return;
    }
    huffman_view_index = 0;
    refreshHuffmanView();
}

void ImageGUI::onHuffmanTreeViewClicked()
{
    if (huffman_highlight_image.empty() || huffman_tree_image.empty()) {
        return;
    }
    huffman_view_index = 1;
    refreshHuffmanView();
}

void ImageGUI::onHuffmanAreaTableCellClicked(int row, int column)
{
    (void)column;

    if (row < 0 || row >= static_cast<int>(huffman_all_label_areas.size())) {
        return;
    }

    huffman_selected_area_label = huffman_all_label_areas[row].first;
    huffman_view_index = 0;
    tab_widget->setCurrentIndex(3);
    refreshHuffmanView();
}

bool ImageGUI::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == huffman_label && event->type() == QEvent::MouseButtonPress) {
        if (huffman_view_index != 1 || huffman_tree_image.empty() || huffman_tree_node_centers.empty()) {
            return false;
        }

        const QPixmap pix = huffman_label->pixmap(Qt::ReturnByValue);
        if (pix.isNull()) {
            return false;
        }

        auto *mouseEvent = static_cast<QMouseEvent *>(event);
        const QSize pixSize = pix.size();
        const QSize labelSize = huffman_label->size();
        const int offsetX = max(0, (labelSize.width() - pixSize.width()) / 2);
        const int offsetY = max(0, (labelSize.height() - pixSize.height()) / 2);

        const QPoint clickPos = mouseEvent->pos();
        const int px = clickPos.x() - offsetX;
        const int py = clickPos.y() - offsetY;
        if (px < 0 || py < 0 || px >= pixSize.width() || py >= pixSize.height()) {
            return false;
        }

        const double sx = static_cast<double>(huffman_tree_image.cols) / static_cast<double>(pixSize.width());
        const double sy = static_cast<double>(huffman_tree_image.rows) / static_cast<double>(pixSize.height());
        const Point mappedPoint(cvRound(px * sx), cvRound(py * sy));

        int hitIndex = -1;
        double bestDist2 = 1e18;
        const double hitRadius = 22.0;
        const double hitRadius2 = hitRadius * hitRadius;
        for (int i = 0; i < static_cast<int>(huffman_tree_node_centers.size()); ++i) {
            const Point c = huffman_tree_node_centers[i];
            const double dx = static_cast<double>(mappedPoint.x - c.x);
            const double dy = static_cast<double>(mappedPoint.y - c.y);
            const double dist2 = dx * dx + dy * dy;
            if (dist2 <= hitRadius2 && dist2 < bestDist2) {
                bestDist2 = dist2;
                hitIndex = i;
            }
        }

        if (hitIndex >= 0) {
            huffman_selected_tree_node_index = hitIndex;
            refreshHuffmanView();
            return true;
        }
    }

    return QMainWindow::eventFilter(watched, event);
}

void ImageGUI::syncSidebarByTab(int index)
{
    if (!sidebar_widget || !sidebar_stack || !sidebar_title_label) {
        return;
    }

    if (index == 0) {
        sidebar_widget->hide();
        return;
    }

    sidebar_widget->show();
    if (index == 1) {
        sidebar_title_label->setText("采样点坐标");
        sidebar_stack->setCurrentIndex(0);
    } else if (index == 2) {
        sidebar_title_label->setText("区域着色信息");
        sidebar_stack->setCurrentIndex(1);
    } else if (index == 3) {
        if (huffman_view_index == 0) {
            sidebar_title_label->setText("区域面积表");
            sidebar_stack->setCurrentIndex(2);
        } else {
            sidebar_title_label->setText("Huffman编码结果");
            sidebar_stack->setCurrentIndex(3);
        }
    } else {
        sidebar_widget->hide();
    }
}
