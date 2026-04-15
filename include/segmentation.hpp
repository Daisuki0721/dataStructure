 #ifndef __SEGMENTATON_HPP__
#define __SEGMENTATON_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include <numeric>
#include <algorithm>

#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <opencv2/opencv.hpp>

class SeedSampler {
public:
    static std::vector<cv::Point> generateSeeds(int width, int height, int k) {
        std::vector<cv::Point> seeds;
        if (width <= 0 || height <= 0 || k <= 0) return seeds;

        // 1. 理论极限半径 (六角堆积密度 ~0.906)
        // 设半径为 R (距离 d = 2R)，面积 A = K * 2 * sqrt(3) * R^2
        // 简化计算：d = sqrt( (2 * Area) / (sqrt(3) * K) )
        double minDist = std::sqrt((2.0 * width * height) / (1.732 * k));
        
        // 稍微缩减 5% 确保随机生长下也能塞进 K 个点
        minDist *= 0.95; 

        const int maxAttempts = 30;
        double minDistSq = minDist * minDist;
        
        // 空间哈希网格
        double cellSize = minDist / std::sqrt(2.0);
        int gridW = static_cast<int>(std::ceil(width / cellSize));
        int gridH = static_cast<int>(std::ceil(height / cellSize));
        std::vector<int> grid(gridW * gridH, -1);

        std::vector<cv::Point> activeList;
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::uniform_real_distribution<double> xDist(0, width - 1);
        std::uniform_real_distribution<double> yDist(0, height - 1);
        std::uniform_real_distribution<double> angleDist(0, 2.0 * CV_PI);

        // 放置首个种子
        cv::Point first(width / 2, height / 2);
        seeds.push_back(first);
        activeList.push_back(first);
        grid[getGridIdx(first, cellSize, gridW)] = 0;

        // 2. 紧凑生长循环
        while (!activeList.empty() && seeds.size() < (size_t)k) {
            // 策略：总是从最新的点或随机点开始生长
            int idxInActive = (int)activeList.size() - 1; 
            cv::Point p = activeList[idxInActive];
            bool found = false;

            for (int i = 0; i < maxAttempts; ++i) {
                // 【核心修改】：距离固定为 minDist + 微小偏置，只随机角度
                double theta = angleDist(rng);
                cv::Point candidate(
                    static_cast<int>(std::round(p.x + (minDist + 0.1) * std::cos(theta))),
                    static_cast<int>(std::round(p.y + (minDist + 0.1) * std::sin(theta)))
                );

                if (candidate.x >= 0 && candidate.x < width && candidate.y >= 0 && candidate.y < height) {
                    if (isFarEnough(candidate, minDistSq, cellSize, gridW, gridH, grid, seeds)) {
                        seeds.push_back(candidate);
                        activeList.push_back(candidate);
                        grid[getGridIdx(candidate, cellSize, gridW)] = (int)seeds.size() - 1;
                        found = true;
                        break; // 成功找到一个，继续从这个新点生长（深度优先感）
                    }
                }
            }

            if (!found) {
                activeList.erase(activeList.begin() + idxInActive);
            }
        }

        // 3. 极端保底：如果还是不够（通常是因为初始半径计算依然偏大）
        if (seeds.size() < (size_t)k) {
            // 降低阈值，填补剩余空间
            double fallbackDistSq = minDistSq * 0.5; 
            for (int i = 0; i < 2000 && seeds.size() < (size_t)k; ++i) {
                cv::Point p(static_cast<int>(xDist(rng)), static_cast<int>(yDist(rng)));
                if (isFarEnough(p, fallbackDistSq, cellSize, gridW, gridH, grid, seeds)) {
                    seeds.push_back(p);
                }
            }
        }

        return seeds;
    }

private:
    static int getGridIdx(const cv::Point& p, double cellSize, int gridW) {
        return (int)(p.y / cellSize) * gridW + (int)(p.x / cellSize);
    }

    static bool isFarEnough(const cv::Point& p, double dSq, double cellSize, 
                            int gridW, int gridH, const std::vector<int>& grid, 
                            const std::vector<cv::Point>& seeds) {
        int cx = (int)(p.x / cellSize);
        int cy = (int)(p.y / cellSize);
        for (int y = std::max(0, cy - 2); y <= std::min(gridH - 1, cy + 2); ++y) {
            for (int x = std::max(0, cx - 2); x <= std::min(gridW - 1, cx + 2); ++x) {
                int idx = grid[y * gridW + x];
                if (idx != -1) {
                    double dx = p.x - seeds[idx].x;
                    double dy = p.y - seeds[idx].y;
                    if (dx * dx + dy * dy < dSq) return false;
                }
            }
        }
        return true;
    }
};

#endif
