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

class SeedSampler {
public:
    static std::vector<cv::Point> generateSeeds(int width, int height, int k) {
        std::vector<cv::Point> seeds;
        if (width <= 0 || height <= 0 || k <= 0) return seeds;

        // 1. 计算理想距离 (基于六角堆积密度)
        // 增加约 10% 的冗余空间，确保能采够 K 个点
        double minDist = std::sqrt((2.0 * width * height) / (1.732 * k)) * 0.92;
        double minDistSq = minDist * minDist;

        // 空间哈希网格加速
        double cellSize = minDist / std::sqrt(2.0);
        int gridW = static_cast<int>(std::ceil(width / cellSize));
        int gridH = static_cast<int>(std::ceil(height / cellSize));
        std::vector<int> grid(gridW * gridH, -1);

        std::vector<cv::Point> activeList;
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        
        // 初始点放在中心
        cv::Point first(width / 2, height / 2);
        seeds.push_back(first);
        activeList.push_back(first);
        grid[getGridIdx(first, cellSize, gridW)] = 0;

        // 预定义六个方向的角度 (弧度)
        const double directions[6] = {
            0.0, 
            M_PI / 3.0,       // 60 deg
            2.0 * M_PI / 3.0, // 120 deg
            M_PI,             // 180 deg
            4.0 * M_PI / 3.0, // 240 deg
            5.0 * M_PI / 3.0  // 300 deg
        };

        // 扰动范围：让生成的晶格不至于死板，且能适应像素对齐
        std::uniform_real_distribution<double> jitterDist(-0.1, 0.1); 
        // 随机打乱方向顺序的索引
        std::vector<int> dirIndices = {0, 1, 2, 3, 4, 5};

        // 2. 迭代采样
        while (!activeList.empty() && seeds.size() < (size_t)k) {
            // 采用“新点优先”策略，有助于快速扩散
            int activeIdx = (int)activeList.size() - 1;
            cv::Point p = activeList[activeIdx];
            bool foundAtLeastOne = false;

            // 随机化搜索方向顺序
            std::shuffle(dirIndices.begin(), dirIndices.end(), rng);

            for (int i : dirIndices) {
                // 方向 = 标准六边形方向 + 微小抖动
                double angle = directions[i] + jitterDist(rng);
                
                // 步进距离 = minDist + 1 (确保跨过严格边界)
                cv::Point candidate(
                    static_cast<int>(std::round(p.x + (minDist + 1.0) * std::cos(angle))),
                    static_cast<int>(std::round(p.y + (minDist + 1.0) * std::sin(angle)))
                );

                if (candidate.x >= 0 && candidate.x < width && candidate.y >= 0 && candidate.y < height) {
                    if (isFarEnough(candidate, minDistSq, cellSize, gridW, gridH, grid, seeds)) {
                        seeds.push_back(candidate);
                        activeList.push_back(candidate);
                        grid[getGridIdx(candidate, cellSize, gridW)] = (int)seeds.size() - 1;
                        foundAtLeastOne = true;
                        
                        // 如果采够了立刻退出
                        if (seeds.size() >= (size_t)k) break;
                    }
                }
            }

            // 如果该点六个方向都尝试过了无法继续生长，则剔除
            if (!foundAtLeastOne) {
                activeList.erase(activeList.begin() + activeIdx);
            }
        }

        // 3. 保底逻辑 (处理极少数由于几何死角没填满的情况)
        if (seeds.size() < (size_t)k) {
            std::uniform_int_distribution<int> xDist(0, width - 1);
            std::uniform_int_distribution<int> yDist(0, height - 1);
            for (int i = 0; i < 5000 && seeds.size() < (size_t)k; ++i) {
                cv::Point p(xDist(rng), yDist(rng));
                if (isFarEnough(p, minDistSq * 0.8, cellSize, gridW, gridH, grid, seeds)) {
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
        // 检查邻域网格
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
