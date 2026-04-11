 #ifndef __SEGMENTATON_HPP__
#define __SEGMENTATON_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <ctime>

class SeedSampler {
public:
    // 根据公式计算最小距离阈值：(M * N / K)^0.5
    static double calculateMinDistance(int width, int height, int k) {
        return std::sqrt(static_cast<double>(width * height) / k);
    }

    // 泊松圆盘采样 + 贪心策略
    static std::vector<cv::Point> generateSeeds(int width, int height, int k) {
        std::vector<cv::Point> seeds;
        double minDist = calculateMinDistance(width, height, k);
        double minDistSq = minDist * minDist; // 使用距离的平方来比较，避免开方运算
        
        // 设置随机种子
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        int attempts = 0;
        int maxAttempts = k * 100; // 贪心尝试上限

        while (seeds.size() < static_cast<size_t>(k) && attempts < maxAttempts) {
            int x = std::rand() % width;
            int y = std::rand() % height;
            cv::Point newPoint(x, y);

            bool farEnough = true;
            for (const auto& existingPoint : seeds) {
                // 计算欧几里得距离的平方，避免开方运算
                double dx = newPoint.x - existingPoint.x;
                double dy = newPoint.y - existingPoint.y;
                if ((dx * dx + dy * dy) < minDistSq) {
                farEnough = false;
                break;
                }
            }

            if (farEnough) {
                seeds.push_back(newPoint);
            }
            attempts++;
        }
        return seeds;
    }
};

#endif
