#include <iostream>
#include <vector>
#include <chrono>
#include "segmentation.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

using namespace cv;
using namespace std;

int main() {
    //读取图像
    Mat src = imread("../fruits.jpg"); 
    if (src.empty()) {
        cout << "错误：无法读取 fruits.jpg，请检查文件路径！" << endl;
        return -1;
    }

    //开始计时
    auto start_time = chrono::high_resolution_clock::now();

    int K = 25; // 种子点数量
    int width = src.cols;
    int height = src.rows;

    //生成均匀随机种子点
    vector<Point> seedPoints = SeedSampler::generateSeeds(width, height, K);

    Mat markers = Mat::zeros(src.size(), CV_32S);
    for (size_t i = 0; i < seedPoints.size(); i++) {
        circle(markers, seedPoints[i], 1, Scalar(static_cast<int>(i + 1)), -1);
    }

    //预处理：均值漂移滤波
    Mat shifted;
    pyrMeanShiftFiltering(src, shifted, 10, 51);
    
    //分水岭
    watershed(shifted, markers);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end_time - start_time;
    cout << "算法执行耗时: " << duration.count() << " 毫秒" << endl;

    vector<Vec3b> colors;
    for (int i = 0; i < K; i++) {
        colors.push_back(Vec3b(rand() & 255, rand() & 255, rand() & 255));
    }

    Mat colorMask = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= K) {
                colorMask.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }

    Mat result;
    addWeighted(src, 0.6, colorMask, 0.4, 0, result);

    for (const auto& pt : seedPoints) {
        circle(result, pt, 3, Scalar(0, 0, 255), -1);
        circle(result, pt, 3, Scalar(255, 255, 255), 1); 
    }

    string timeText = "Time: " + to_string(duration.count()) + " ms";
    putText(result, timeText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    //显示最终合成结果
    imshow("Task 1: Seed & Watershed Result", result);
    
    cout << "按下任意键退出程序..." << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}