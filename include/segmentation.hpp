 #ifndef __SEGMENTATON_HPP__
#define __SEGMENTATON_HPP__

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include <numeric>
#include <algorithm>

class SeedSampler {
public:
    // 根据公式计算最小距离阈值：(M * N / K)^0.5
    static double calculateMinDistance(int width, int height, int k) {
        return std::sqrt(static_cast<double>(width * height) / k);
    }

    // PDS(泊松圆盘) + FPS(最远点采样)：更接近六边形均匀分布的受限随机采样
    static std::vector<cv::Point> generateSeeds(int width, int height, int k) {
        std::vector<cv::Point> seeds;

        if (width <= 0 || height <= 0 || k <= 0) {
            return seeds;
        }

        const int totalPixels = width * height;
        const int target = std::min(k, totalPixels);
        seeds.reserve(static_cast<size_t>(target));

        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::uniform_int_distribution<int> xDist(0, width - 1);
        std::uniform_int_distribution<int> yDist(0, height - 1);

        const double minDist = std::max(1.0, calculateMinDistance(width, height, std::max(1, target)));
        const double minDistSq = minDist * minDist;

        // Bridson 常用设置：cellSize = r / sqrt(2)
        const int cellSize = std::max(1, static_cast<int>(std::floor(minDist / std::sqrt(2.0))));
        const int gridW = (width + cellSize - 1) / cellSize;
        const int gridH = (height + cellSize - 1) / cellSize;

        auto toCellIndex = [&](const cv::Point& p) {
            int cx = std::min(gridW - 1, std::max(0, p.x / cellSize));
            int cy = std::min(gridH - 1, std::max(0, p.y / cellSize));
            return cy * gridW + cx;
        };

        auto farEnough = [&](const cv::Point& p,
                             const std::vector<cv::Point>& points,
                             const std::vector<std::vector<int>>& buckets) -> bool {
            int cx = std::min(gridW - 1, std::max(0, p.x / cellSize));
            int cy = std::min(gridH - 1, std::max(0, p.y / cellSize));
            int neighborRange = static_cast<int>(std::ceil(minDist / cellSize));

            int minY = std::max(0, cy - neighborRange);
            int maxY = std::min(gridH - 1, cy + neighborRange);
            int minX = std::max(0, cx - neighborRange);
            int maxX = std::min(gridW - 1, cx + neighborRange);

            for (int yy = minY; yy <= maxY; ++yy) {
                for (int xx = minX; xx <= maxX; ++xx) {
                    const auto& cell = buckets[static_cast<size_t>(yy * gridW + xx)];
                    for (int idx : cell) {
                        const cv::Point& q = points[static_cast<size_t>(idx)];
                        double dx = static_cast<double>(p.x - q.x);
                        double dy = static_cast<double>(p.y - q.y);
                        // 严格大于阈值
                        if ((dx * dx + dy * dy) <= minDistSq) {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        // 1) PDS 生成蓝噪声候选集（受限随机，天然趋向蜂窝/六边形稳定态）
        std::vector<cv::Point> candidates;
        candidates.reserve(static_cast<size_t>(std::max(target * 2, target + 64)));
        std::vector<std::vector<int>> buckets(static_cast<size_t>(gridW * gridH));
        std::vector<int> active;

        auto addCandidate = [&](const cv::Point& p) {
            int idx = static_cast<int>(candidates.size());
            candidates.push_back(p);
            buckets[static_cast<size_t>(toCellIndex(p))].push_back(idx);
            active.push_back(idx);
        };

        addCandidate(cv::Point(xDist(rng), yDist(rng)));

        const int kTrials = 30;
        const int candidateCap = std::max(target * 3, target + 128);

        while (!active.empty() && static_cast<int>(candidates.size()) < candidateCap) {
            std::uniform_int_distribution<int> activeDist(0, static_cast<int>(active.size()) - 1);
            int activePos = activeDist(rng);
            int baseIdx = active[static_cast<size_t>(activePos)];
            cv::Point base = candidates[static_cast<size_t>(baseIdx)];

            bool found = false;
            for (int t = 0; t < kTrials && static_cast<int>(candidates.size()) < candidateCap; ++t) {
                double angle = 2.0 * CV_PI * unit(rng);
                // 在 [r, 2r] 环带内采样
                double radius = minDist * (1.0 + unit(rng));
                int nx = static_cast<int>(std::round(base.x + radius * std::cos(angle)));
                int ny = static_cast<int>(std::round(base.y + radius * std::sin(angle)));

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                cv::Point p(nx, ny);
                if (!farEnough(p, candidates, buckets)) {
                    continue;
                }

                addCandidate(p);
                found = true;
                break;
            }

            if (!found) {
                active[static_cast<size_t>(activePos)] = active.back();
                active.pop_back();
            }
        }

        // 2) 若候选仍不足，再做严格距离的随机补点（不放宽阈值）
        int fallbackAttempts = 0;
        int fallbackLimit = std::max(2000, target * 80);
        while (static_cast<int>(candidates.size()) < target && fallbackAttempts < fallbackLimit) {
            cv::Point p(xDist(rng), yDist(rng));
            if (farEnough(p, candidates, buckets)) {
                int idx = static_cast<int>(candidates.size());
                candidates.push_back(p);
                buckets[static_cast<size_t>(toCellIndex(p))].push_back(idx);
            }
            ++fallbackAttempts;
        }

        if (candidates.empty()) {
            return seeds;
        }

        // 3) 如果候选数>=K，使用 FPS 选 K 个，增强全局均匀性
        if (static_cast<int>(candidates.size()) > target) {
            std::vector<double> bestDistSq(candidates.size(), std::numeric_limits<double>::max());
            std::vector<unsigned char> chosen(candidates.size(), 0);

            std::uniform_int_distribution<int> firstDist(0, static_cast<int>(candidates.size()) - 1);
            int first = firstDist(rng);
            chosen[static_cast<size_t>(first)] = 1;
            seeds.push_back(candidates[static_cast<size_t>(first)]);

            for (size_t i = 0; i < candidates.size(); ++i) {
                double dx = static_cast<double>(candidates[i].x - candidates[static_cast<size_t>(first)].x);
                double dy = static_cast<double>(candidates[i].y - candidates[static_cast<size_t>(first)].y);
                bestDistSq[i] = dx * dx + dy * dy;
            }

            while (static_cast<int>(seeds.size()) < target) {
                int pick = -1;
                double maxMinDist = -1.0;

                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (chosen[i]) {
                        continue;
                    }
                    if (bestDistSq[i] > maxMinDist) {
                        maxMinDist = bestDistSq[i];
                        pick = static_cast<int>(i);
                    }
                }

                if (pick < 0) {
                    break;
                }

                chosen[static_cast<size_t>(pick)] = 1;
                const cv::Point pickedPoint = candidates[static_cast<size_t>(pick)];
                seeds.push_back(pickedPoint);

                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (chosen[i]) {
                        continue;
                    }
                    double dx = static_cast<double>(candidates[i].x - pickedPoint.x);
                    double dy = static_cast<double>(candidates[i].y - pickedPoint.y);
                    double d2 = dx * dx + dy * dy;
                    if (d2 < bestDistSq[i]) {
                        bestDistSq[i] = d2;
                    }
                }
            }

            return seeds;
        }

        // 候选数不足K时，返回全部严格合法点（全部互异且满足距离阈值）
        return candidates;
    }
};

#endif
