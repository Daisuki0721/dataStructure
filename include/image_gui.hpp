#ifndef IMAGE_GUI_HPP
#define IMAGE_GUI_HPP

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QComboBox>
#include <QPoint>
#include <opencv2/opencv.hpp>
#include <QObject>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

class QTabWidget;
class QTableWidget;
class QStackedWidget;
class QScrollArea;

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
    void startTask3HuffmanSort();
    void saveResult();
    void onTabChanged(int index);
    void onViewModeChanged(int index);
    void onSeedTableCellClicked(int row, int column);
    void onInfoTableCellClicked(int row, int column);
    void onZoomPercentChanged(int value);
    void onFitLockToggled();
    void onHuffmanAreaViewClicked();
    void onHuffmanTreeViewClicked();
    void onHuffmanAreaTableCellClicked(int row, int column);
    void onHuffmanCodeTableCellClicked(int row, int column);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

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
    void updateHuffmanTable(const std::vector<std::pair<int, int>> &allLabelAreas,
                            const std::set<int> &matchedLabels);
    void updateHuffmanCodeTable(const std::vector<std::tuple<int, int, std::string>> &codeRows);
    void refreshCurrentView();
    void refreshHuffmanView();
    QScrollArea *scrollAreaForLabel(const QLabel *label) const;
    bool handleHuffmanTreeNodeHit(const QPoint &labelPos);
    void syncSidebarByTab(int index);

    QLabel *original_label;
    QLabel *result_label;
    QLabel *coloring_label;
    QLabel *huffman_label;
    QLabel *huffman_view_title_label;
    QLabel *sidebar_title_label;
    QLabel *file_label;
    QLabel *status_label;
    QPushButton *file_button;
    QPushButton *segment_button;
    QPushButton *task2_button;
    QPushButton *task3_button;
    QPushButton *fit_lock_button;
    QPushButton *huffman_area_button;
    QPushButton *huffman_tree_button;
    QPushButton *save_button;
    QSpinBox *k_spinbox;
    QSpinBox *zoom_spinbox;
    QComboBox *view_mode_combo;
    QTabWidget *tab_widget;
    QScrollArea *original_scroll_area;
    QScrollArea *result_scroll_area;
    QScrollArea *coloring_scroll_area;
    QScrollArea *huffman_scroll_area;
    QWidget *sidebar_widget;
    QStackedWidget *sidebar_stack;
    QTableWidget *seed_table;
    QTableWidget *info_table;
    QTableWidget *huffman_table;
    QTableWidget *huffman_code_table;

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

    cv::Mat huffman_highlight_image;
    cv::Mat huffman_tree_image;
    cv::Mat huffman_markers;
    int huffman_view_index;
    int huffman_selected_area_label;
    int huffman_selected_tree_node_index;
    int huffman_last_low;
    int huffman_last_high;
    int zoom_percent;
    bool fit_lock_enabled;
    bool drag_panning_active;
    bool drag_panning_moved;
    QPoint drag_last_pos;
    QScrollArea *drag_scroll_area;
    QLabel *drag_label;
    std::vector<cv::Point> huffman_tree_node_centers;
    std::vector<int> huffman_tree_node_labels;
    std::unordered_map<int, int> huffman_code_row_by_label;
    std::vector<std::pair<int, int>> huffman_all_label_areas;
    std::set<int> huffman_matched_labels;
    std::vector<std::tuple<int, int, std::string>> huffman_code_rows;

    std::string image_path;
    int k_value;
};

#endif // IMAGE_GUI_HPP
