#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "four_colors.hpp"
#include "segmentation.hpp"

namespace {

struct RunResult {
    bool success = false;
    bool writeOk = true;
    double elapsedMs = 0.0;
    int seedCount = 0;
    int regionCount = 0;
    size_t edgeCount = 0;
    int maxDegree = 0;
    int backtracks = 0;
};

RunResult runTask2(const cv::Mat& src, int k, const std::string& outputPath, bool writeImage) {
    RunResult r;
    auto t0 = std::chrono::high_resolution_clock::now();

    const int width = src.cols;
    const int height = src.rows;

    std::vector<cv::Point> seedPoints = SeedSampler::generateSeeds(width, height, k);
    r.seedCount = static_cast<int>(seedPoints.size());
    if (seedPoints.empty()) {
        return r;
    }

    cv::Mat markers = cv::Mat::zeros(src.size(), CV_32S);
    for (int i = 0; i < r.seedCount; ++i) {
        cv::circle(markers, seedPoints[i], 1, cv::Scalar(i + 1), -1);
    }

    cv::Mat shifted;
    cv::pyrMeanShiftFiltering(src, shifted, 10, 51);
    cv::watershed(shifted, markers);

    std::vector<int> regionLabels;
    std::unordered_map<int, int> labelToNode;
    fourcolor::AdjList adjacency = fourcolor::buildAdjacencyFromWatershed(markers, regionLabels, labelToNode);
    const std::vector<int> bfsOrder = fourcolor::buildBfsOrder(adjacency);

    std::vector<int> nodeColors;
    fourcolor::ColoringOptions options;
    if (k >= 1000) {
        options.maxBacktracks = 450000;
        options.maxMs = 3600.0;
    } else if (k >= 500) {
        options.maxBacktracks = 250000;
        options.maxMs = 1800.0;
    } else {
        options.maxBacktracks = 120000;
        options.maxMs = 450.0;
    }
    const fourcolor::ColoringStats cstats =
        fourcolor::colorWithBfsStackBacktracking(adjacency, bfsOrder, nodeColors, options);

    r.success = cstats.success;
    r.regionCount = static_cast<int>(regionLabels.size());
    r.backtracks = cstats.backtracks;

    for (const auto& nbs : adjacency) {
        r.edgeCount += nbs.size();
        r.maxDegree = std::max(r.maxDegree, static_cast<int>(nbs.size()));
    }
    r.edgeCount /= 2;

    if (r.success && writeImage) {
        const std::array<cv::Vec3b, 4> palette = {
            cv::Vec3b(50, 50, 230),
            cv::Vec3b(50, 180, 50),
            cv::Vec3b(230, 80, 40),
            cv::Vec3b(40, 220, 220)
        };

        cv::Mat recolored = cv::Mat::zeros(src.size(), CV_8UC3);
        for (int y = 0; y < markers.rows; ++y) {
            for (int x = 0; x < markers.cols; ++x) {
                const int label = markers.at<int>(y, x);
                if (label == -1) {
                    recolored.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
                } else if (label > 0) {
                    auto it = labelToNode.find(label);
                    if (it != labelToNode.end()) {
                        const int node = it->second;
                        recolored.at<cv::Vec3b>(y, x) = palette[nodeColors[node]];
                    }
                }
            }
        }

        cv::Mat blended;
        cv::addWeighted(src, 0.55, recolored, 0.45, 0.0, blended);
        r.writeOk = cv::imwrite(outputPath, blended);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

void benchmark(const cv::Mat& src, int trials) {
    struct Requirement {
        int k;
        double maxFailureRate;
        double maxMs;
        char level;
    };

    const std::array<Requirement, 3> reqs = {
        Requirement{1000, 0.40, 4000.0, 'A'},
        Requirement{500, 0.20, 2000.0, 'B'},
        Requirement{100, 0.10, 500.0, 'C'}
    };

    std::cout << "Benchmark 开始，trials=" << trials << "\n";
    for (const auto& req : reqs) {
        int failCount = 0;
        int overtimeCount = 0;
        double sumMs = 0.0;
        double worstMs = 0.0;
        double sumBacktracks = 0.0;

        for (int i = 0; i < trials; ++i) {
            const RunResult rr = runTask2(src, req.k, "", false);
            const bool overtime = rr.elapsedMs > req.maxMs;
            const bool fail = (!rr.success) || overtime;

            if (!rr.success) {
                // Keep for observability; overall failure is counted by fail below.
            }
            if (overtime) {
                overtimeCount++;
            }

            sumMs += rr.elapsedMs;
            worstMs = std::max(worstMs, rr.elapsedMs);
            sumBacktracks += rr.backtracks;

            if (fail) {
                failCount++;
            }
        }

        const double failureRate = static_cast<double>(failCount) / static_cast<double>(trials);
        const double avgMs = sumMs / static_cast<double>(trials);
        const double avgBacktracks = sumBacktracks / static_cast<double>(trials);
        const bool pass = failureRate < req.maxFailureRate && worstMs <= req.maxMs;

        std::cout << "[等级 " << req.level << "] K=" << req.k << "\n";
        std::cout << "  失败率(含超时): " << failureRate * 100.0 << "% (要求 < "
                  << req.maxFailureRate * 100.0 << "%)\n";
        std::cout << "  最慢耗时: " << worstMs << " ms (要求 <= " << req.maxMs << " ms)\n";
        std::cout << "  平均耗时: " << avgMs << " ms\n";
        std::cout << "  平均回溯次数: " << avgBacktracks << "\n";
        std::cout << "  超时次数: " << overtimeCount << "\n";
        std::cout << "  结果: " << (pass ? "PASS" : "FAIL") << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "用法: ./task2 <input_image> <output_image> [K]\n";
        std::cout << "示例: ./task2 ../assets/fruits.jpg ../outputs/task2_colored.png 120\n";
        std::cout << "评测: ./task2 --benchmark <input_image> [trials]\n";
        return 1;
    }

    if (std::string(argv[1]) == "--benchmark") {
        if (argc < 3) {
            std::cout << "用法: ./task2 --benchmark <input_image> [trials]\n";
            return 1;
        }

        const std::string inputPath = argv[2];
        const int trials = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 20;

        cv::Mat src = cv::imread(inputPath);
        if (src.empty()) {
            std::cerr << "错误: 无法读取输入图像: " << inputPath << "\n";
            return 1;
        }

        benchmark(src, trials);
        return 0;
    }

    if (argc < 3) {
        std::cout << "用法: ./task2 <input_image> <output_image> [K]\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];
    const int K = (argc >= 4) ? std::max(4, std::atoi(argv[3])) : 120;

    cv::Mat src = cv::imread(inputPath);
    if (src.empty()) {
        std::cerr << "错误: 无法读取输入图像: " << inputPath << "\n";
        return 1;
    }

    const RunResult rr = runTask2(src, K, outputPath, true);
    if (!rr.success) {
        std::cerr << "错误: 四原色回溯着色失败或图着色非法。\n";
        return 1;
    }

    if (!rr.writeOk) {
        std::cerr << "错误: 输出图像写入失败: " << outputPath << "\n";
        return 1;
    }

    std::cout << "Task2 完成\n";
    std::cout << "输入图像: " << inputPath << "\n";
    std::cout << "输出图像: " << outputPath << "\n";
    std::cout << "目标种子数 K: " << K << "，实际种子数: " << rr.seedCount << "\n";
    std::cout << "区域数: " << rr.regionCount << "，邻接边数: " << rr.edgeCount
              << "，最大度: " << rr.maxDegree << "\n";
    std::cout << "回溯次数: " << rr.backtracks << "\n";
    std::cout << "总耗时: " << rr.elapsedMs << " ms\n";

    return 0;
}
