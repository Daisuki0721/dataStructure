 #ifndef __SEGMENTATON_HPP__
#define __SEGMENTATON_HPP__

#include <opencv2/opencv.hpp>
#include <ctime>
#include <random>
#include <numeric>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

class SeedSampler {
public:
    /**
     * @param k         目标种子数
     * @param jitter    偏移扰动范围 (0.0 为严格六角网格, >0 为在格点内的随机偏移比例)
     */
    static std::vector<cv::Point> generateSeeds(int width, int height, int k, double jitter = 0.0) {
        std::vector<cv::Point> seeds;
        if (width <= 0 || height <= 0 || k <= 0) return seeds;

        // 1. 根据 K 计算最优的六角网格步长 baseDist
        // 面积公式: Area = (sqrt(3)/2) * baseDist^2 * K
        double baseDist = std::sqrt((2.0 * width * height) / (std::sqrt(3.0) * k));
        
        double dx = baseDist;
        double dy = baseDist * std::sqrt(3.0) / 2.0;

        // 2. 生成覆盖全图的候选格点
        struct Candidate {
            cv::Point2f pos;
            float distToBorder;
        };
        std::vector<Candidate> candidates;

        int rowCount = static_cast<int>(std::ceil(height / dy)) + 1;
        int colCount = static_cast<int>(std::ceil(width / dx)) + 1;

        for (int r = 0; r < rowCount; ++r) {
            double xOffset = (r % 2 == 0) ? 0 : dx * 0.5;
            for (int c = 0; c < colCount; ++c) {
                float x = static_cast<float>(c * dx + xOffset);
                float y = static_cast<float>(r * dy);

                // 只保留图像内的点
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    // 计算该点到四条边的最小距离
                    float dLeft = x;
                    float dRight = (width - 1) - x;
                    float dTop = y;
                    float dBottom = (height - 1) - y;
                    float minDistToBorder = std::min({dLeft, dRight, dTop, dBottom});

                    candidates.push_back({{x, y}, minDistToBorder});
                }
            }
        }

        // 3. 排序：从小到大排序（边缘优先）
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.distToBorder < b.distToBorder;
        });

        // 4. 选取前 K 个点并添加随机扰动 (Jitter)
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::uniform_real_distribution<double> offsetDist(-jitter * baseDist * 0.5, jitter * baseDist * 0.5);

        size_t actualK = std::min(candidates.size(), static_cast<size_t>(k));
        seeds.reserve(actualK);

        for (size_t i = 0; i < actualK; ++i) {
            double ox = (jitter > 0) ? offsetDist(rng) : 0;
            double oy = (jitter > 0) ? offsetDist(rng) : 0;

            int finalX = std::clamp(static_cast<int>(std::round(candidates[i].pos.x + ox)), 0, width - 1);
            int finalY = std::clamp(static_cast<int>(std::round(candidates[i].pos.y + oy)), 0, height - 1);
            
            seeds.push_back(cv::Point(finalX, finalY));
        }

        return seeds;
    }
};


#endif
