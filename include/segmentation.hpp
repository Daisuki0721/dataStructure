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

    // 多相位六角晶格 + FPS：在严格距离约束下尽量逼近最密均匀分布
    static std::vector<cv::Point> generateSeeds(int width, int height, int k) {
        std::vector<cv::Point> seeds;

        if (width <= 0 || height <= 0 || k <= 0) {
            return seeds;
        }

        const int totalPixels = width * height;
        const int target = std::min(k, totalPixels);
        seeds.reserve(static_cast<size_t>(target));

        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        std::uniform_int_distribution<int> xDist(0, width - 1);
        std::uniform_int_distribution<int> yDist(0, height - 1);

        const double minDist = std::max(1.0, calculateMinDistance(width, height, std::max(1, target)));
        const double minDistSq = minDist * minDist;

        // 用于严格距离验证的空间哈希桶。
        const int cellSize = std::max(1, static_cast<int>(std::floor(minDist / std::sqrt(2.0))));
        const int gridW = (width + cellSize - 1) / cellSize;
        const int gridH = (height + cellSize - 1) / cellSize;

        auto toCell = [&](const cv::Point& p) {
            int cx = std::min(gridW - 1, std::max(0, p.x / cellSize));
            int cy = std::min(gridH - 1, std::max(0, p.y / cellSize));
            return std::pair<int, int>(cx, cy);
        };

        auto farEnough = [&](const cv::Point& p,
                             const std::vector<cv::Point>& points,
                             const std::vector<std::vector<int>>& buckets) -> bool {
            auto [cx, cy] = toCell(p);
            int neighborRange = static_cast<int>(std::ceil(minDist / cellSize));

            for (int yy = std::max(0, cy - neighborRange); yy <= std::min(gridH - 1, cy + neighborRange); ++yy) {
                for (int xx = std::max(0, cx - neighborRange); xx <= std::min(gridW - 1, cx + neighborRange); ++xx) {
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

        auto buildHexCandidatesForPhase = [&](double ox, double oy) {
            std::vector<cv::Point> points;
            std::vector<std::vector<int>> buckets(static_cast<size_t>(gridW * gridH));
            std::vector<unsigned char> used(static_cast<size_t>(totalPixels), 0);

            const double dx = std::floor(minDist) + 1.0;
            const double dy = dx * std::sqrt(3.0) * 0.5;

            int row = 0;
            for (double yf = oy; yf < height; yf += dy, ++row) {
                double xStart = ox + ((row & 1) ? dx * 0.5 : 0.0);
                for (double xf = xStart; xf < width; xf += dx) {
                    int x = static_cast<int>(std::round(xf));
                    int y = static_cast<int>(std::round(yf));
                    if (x < 0 || x >= width || y < 0 || y >= height) {
                        continue;
                    }

                    int idxPixel = y * width + x;
                    if (used[static_cast<size_t>(idxPixel)]) {
                        continue;
                    }

                    cv::Point p(x, y);
                    if (!farEnough(p, points, buckets)) {
                        continue;
                    }

                    int idx = static_cast<int>(points.size());
                    points.push_back(p);
                    used[static_cast<size_t>(idxPixel)] = 1;
                    auto [cx, cy] = toCell(p);
                    buckets[static_cast<size_t>(cy * gridW + cx)].push_back(idx);
                }
            }

            return points;
        };

        // 1) 多相位搜索最佳六角晶格偏移，尽量提高可行点上限。
        std::vector<cv::Point> candidates;
        const double dx = std::floor(minDist) + 1.0;
        const double dy = dx * std::sqrt(3.0) * 0.5;
        const int phaseSteps = 8;
        for (int py = 0; py < phaseSteps; ++py) {
            for (int px = 0; px < phaseSteps; ++px) {
                double ox = (static_cast<double>(px) / phaseSteps) * dx;
                double oy = (static_cast<double>(py) / phaseSteps) * dy;
                auto phasePoints = buildHexCandidatesForPhase(ox, oy);
                if (phasePoints.size() > candidates.size()) {
                    candidates.swap(phasePoints);
                }
            }
        }

        // 2) 极端情况下仍不足时，做严格随机补点（不放宽阈值）。
        if (static_cast<int>(candidates.size()) < target) {
            std::vector<std::vector<int>> buckets(static_cast<size_t>(gridW * gridH));
            std::vector<unsigned char> used(static_cast<size_t>(totalPixels), 0);
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                auto [cx, cy] = toCell(candidates[static_cast<size_t>(i)]);
                buckets[static_cast<size_t>(cy * gridW + cx)].push_back(i);
                int idxPixel = candidates[static_cast<size_t>(i)].y * width + candidates[static_cast<size_t>(i)].x;
                used[static_cast<size_t>(idxPixel)] = 1;
            }

            int attempts = 0;
            int limit = std::max(5000, target * 100);
            while (static_cast<int>(candidates.size()) < target && attempts < limit) {
                cv::Point p(xDist(rng), yDist(rng));
                int idxPixel = p.y * width + p.x;
                if (used[static_cast<size_t>(idxPixel)]) {
                    ++attempts;
                    continue;
                }
                if (farEnough(p, candidates, buckets)) {
                    int idx = static_cast<int>(candidates.size());
                    candidates.push_back(p);
                    used[static_cast<size_t>(idxPixel)] = 1;
                    auto [cx, cy] = toCell(p);
                    buckets[static_cast<size_t>(cy * gridW + cx)].push_back(idx);
                }
                ++attempts;
            }
        }

        if (candidates.empty()) {
            return seeds;
        }

        // 3) 候选过多时，用 FPS 选出 K 个，保持全局均匀。
        if (static_cast<int>(candidates.size()) > target) {
            std::vector<double> bestDistSq(candidates.size(), std::numeric_limits<double>::max());
            std::vector<unsigned char> chosen(candidates.size(), 0);

            std::uniform_int_distribution<int> firstDist(0, static_cast<int>(candidates.size()) - 1);
            int first = firstDist(rng);
            chosen[static_cast<size_t>(first)] = 1;
            seeds.push_back(candidates[static_cast<size_t>(first)]);

            for (size_t i = 0; i < candidates.size(); ++i) {
                double dx0 = static_cast<double>(candidates[i].x - candidates[static_cast<size_t>(first)].x);
                double dy0 = static_cast<double>(candidates[i].y - candidates[static_cast<size_t>(first)].y);
                bestDistSq[i] = dx0 * dx0 + dy0 * dy0;
            }

            while (static_cast<int>(seeds.size()) < target) {
                int pick = -1;
                double score = -1.0;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (chosen[i]) {
                        continue;
                    }
                    if (bestDistSq[i] > score) {
                        score = bestDistSq[i];
                        pick = static_cast<int>(i);
                    }
                }
                if (pick < 0) {
                    break;
                }

                chosen[static_cast<size_t>(pick)] = 1;
                const cv::Point chosenPoint = candidates[static_cast<size_t>(pick)];
                seeds.push_back(chosenPoint);

                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (chosen[i]) {
                        continue;
                    }
                    double dxi = static_cast<double>(candidates[i].x - chosenPoint.x);
                    double dyi = static_cast<double>(candidates[i].y - chosenPoint.y);
                    double d2 = dxi * dxi + dyi * dyi;
                    if (d2 < bestDistSq[i]) {
                        bestDistSq[i] = d2;
                    }
                }
            }

            return seeds;
        }

        return candidates;
    }
};

#endif
