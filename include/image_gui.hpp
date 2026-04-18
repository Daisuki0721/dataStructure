#ifndef IMAGE_GUI_HPP
#define IMAGE_GUI_HPP

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class QTabWidget;
class QTableWidget;
class QStackedWidget;

class ImageGUI : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageGUI(QWidget *parent = nullptr);
    ~ImageGUI();

private slots:
    void selectImage();
    void onKValueChanged(int value);
    void startSegmentation();
    void startTask2Recolor();
    void saveResult();
    void onTabChanged(int index);
    void onViewModeChanged(int index);
    void onSeedTableCellClicked(int row, int column);
    void onInfoTableCellClicked(int row, int column);

private:
    void initUI();
    QWidget *createControlPanel();
    void displayImage(const cv::Mat &cv_img, QLabel *label);
    void updateSeedTable(const std::vector<cv::Point> &seedPoints);
    void updateRegionInfoTable(const std::vector<int> &regionLabels,
                               const std::vector<int> &nodeColors,
                               const std::vector<std::vector<int>> &adjacency);
    cv::Mat brightenRegionByLabel(const cv::Mat &baseImage,
                                  const cv::Mat &markers,
                                  int targetLabel);
    void syncSidebarByTab(int index);

    QLabel *original_label;
    QLabel *result_label;
    QLabel *coloring_label;
    QLabel *sidebar_title_label;
    QLabel *file_label;
    QLabel *status_label;
    QPushButton *file_button;
    QPushButton *segment_button;
    QPushButton *task2_button;
    QPushButton *save_button;
    QSpinBox *k_spinbox;
    QComboBox *view_mode_combo;
    QTabWidget *tab_widget;
    QWidget *sidebar_widget;
    QStackedWidget *sidebar_stack;
    QTableWidget *seed_table;
    QTableWidget *info_table;

    cv::Mat original_image;
    cv::Mat result_image;
    cv::Mat task1_cached_image;
    cv::Mat task1_display_image;
    std::vector<cv::Point> current_seed_points;
    cv::Mat task2_base_image;
    cv::Mat task2_display_image;
    cv::Mat task2_markers;
    std::vector<int> task2_region_labels;
    std::vector<int> task2_node_colors;
    std::vector<std::vector<int>> task2_adjacency;
    std::string image_path;
    int k_value;
};

#endif // IMAGE_GUI_HPP
