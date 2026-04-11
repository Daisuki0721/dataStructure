#ifndef IMAGE_GUI_HPP
#define IMAGE_GUI_HPP

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <opencv2/opencv.hpp>
#include <string>

class QTabWidget;

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
    void saveResult();

private:
    void initUI();
    QWidget *createControlPanel();
    void displayImage(const cv::Mat &cv_img, QLabel *label);

    QLabel *original_label;
    QLabel *result_label;
    QLabel *file_label;
    QLabel *status_label;
    QPushButton *file_button;
    QPushButton *segment_button;
    QPushButton *save_button;
    QSpinBox *k_spinbox;
    QTabWidget *tab_widget;

    cv::Mat original_image;
    cv::Mat result_image;
    std::string image_path;
    int k_value;
};

#endif // IMAGE_GUI_HPP
