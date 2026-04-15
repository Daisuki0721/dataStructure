 #ifndef __SEGMENTATON_HPP__
#define __SEGMENTATON_HPP__

#include <opencv2/opencv.hpp>
#include <random>
#include <numeric>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <algorithm>

class SeedSampler {
public:
    /**
     * @param k           参考点数
     * @param max_ratio   扰动倍率上限（如 1.1）
     */
    static std::vector<cv::Point> generateSeeds(int width, int height, int k, double max_ratio = 1.05) {
        std::vector<cv::Point> seeds;
        if (width <= 0 || height <= 0 || k <= 0) return seeds;

        // 1. 基础间距 d 与 空间哈希准备
        double avg_ratio = (1.0 + max_ratio) / 2.0;
        double d = std::sqrt((2.0 * width * height) / (std::sqrt(3.0) * k)) / avg_ratio;
        double dSq = d * d; // 预平方用于快速比较
        
        double base_dy = d * std::sqrt(3.0) / 2.0;

        // 空间哈希桶：格子大小设为 d，保证每个格子里点极少
        double cellSize = d;
        int gridW = static_cast<int>(std::ceil(width / cellSize));
        int gridH = static_cast<int>(std::ceil(height / cellSize));
        std::vector<std::vector<int>> spatialGrid(gridW * gridH);

        // 2. 内部辅助函数
        auto getGridIdx = [&](const cv::Point& p) {
            return (p.y / (int)cellSize) * gridW + (p.x / (int)cellSize);
        };

        auto isSafe = [&](const cv::Point& p) {
            int cx = p.x / (int)cellSize;
            int cy = p.y / (int)cellSize;
            
            // 检查周围 3x3 邻域桶
            for (int y = std::max(0, cy - 1); y <= std::min(gridH - 1, cy + 1); ++y) {
                for (int x = std::max(0, cx - 1); x <= std::min(gridW - 1, cx + 1); ++x) {
                    for (int otherIdx : spatialGrid[y * gridW + x]) {
                        const cv::Point& otherP = seeds[otherIdx];
                        double dx = p.x - otherP.x;
                        double dy = p.y - otherP.y;
                        if ((dx * dx + dy * dy) * 1.25 < dSq) return false;
                    }
                }
            }
            return true;
        };

        // 3. 随机数生成器
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::uniform_real_distribution<double> distRatio(1.0, max_ratio);

        // 4. 逐行生成（由外向内感）
        int row = 0;
        for (double row_y = 0; row_y < height; row_y += base_dy * max_ratio, ++row) {
            
            double current_x = (row % 2 == 1) ? (d * 0.5) : 0.0;

            while (current_x < width) {
                // 每个点产生独立的 dy 和 dx 扰动（仅向上/正向）
                double r_x = distRatio(rng);
                double r_y = distRatio(rng);

                cv::Point candidate(
                    static_cast<int>(std::round(current_x)), // x轴由步进控制
                    static_cast<int>(std::round(row_y + base_dy * (r_y - 1.0))) // y轴向上抖动
                );

                // 边界与安全检查
                if (candidate.x >= 0 && candidate.x < width && candidate.y >= 0 && candidate.y < height) {
                    if (isSafe(candidate)) {
                        spatialGrid[getGridIdx(candidate)].push_back((int)seeds.size());
                        seeds.push_back(candidate);
                    }
                }

                // 步进：基础距离 * 随机扰动，确保下一个点的基础位置足够远
                current_x += d * r_x;
            }
        }

        return seeds;
    }
};

#endif
