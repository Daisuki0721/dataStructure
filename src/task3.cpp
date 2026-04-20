#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "segmentation.hpp"
#include "sort_Huffman.hpp"

namespace fs = std::filesystem;

namespace task3 {

std::vector<RegionArea> buildRegionAreas(const cv::Mat& markers) {
    std::unordered_map<int, int> areaByLabel;
    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            const int label = row[x];
            if (label > 0) {
                areaByLabel[label]++;
            }
        }
    }

    std::vector<RegionArea> areas;
    areas.reserve(areaByLabel.size());
    for (const auto& kv : areaByLabel) {
        areas.push_back(RegionArea{kv.first, kv.second});
    }
    return areas;
}

static void heapify(std::vector<RegionArea>& arr, int n, int i) {
    int largest = i;
    const int left = 2 * i + 1;
    const int right = 2 * i + 2;

    if (left < n && arr[left].area > arr[largest].area) {
        largest = left;
    }
    if (right < n && arr[right].area > arr[largest].area) {
        largest = right;
    }

    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSortByAreaAscending(std::vector<RegionArea>& areas) {
    const int n = static_cast<int>(areas.size());
    for (int i = n / 2 - 1; i >= 0; --i) {
        heapify(areas, n, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        std::swap(areas[0], areas[i]);
        heapify(areas, i, 0);
    }
}

std::pair<int, int> findAreaRangeBinarySearch(const std::vector<RegionArea>& sortedAreasAsc,
                                              int low,
                                              int high) {
    if (sortedAreasAsc.empty() || low > high) {
        return {-1, -1};
    }

    int left = 0;
    int right = static_cast<int>(sortedAreasAsc.size()) - 1;
    int first = -1;
    while (left <= right) {
        const int mid = left + (right - left) / 2;
        if (sortedAreasAsc[mid].area >= low) {
            first = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    left = 0;
    right = static_cast<int>(sortedAreasAsc.size()) - 1;
    int last = -1;
    while (left <= right) {
        const int mid = left + (right - left) / 2;
        if (sortedAreasAsc[mid].area <= high) {
            last = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (first == -1 || last == -1 || first > last) {
        return {-1, -1};
    }
    return {first, last};
}

std::unordered_map<int, cv::Point> computeRegionCentroids(const cv::Mat& markers,
                                                          const std::vector<RegionArea>& selected) {
    std::unordered_set<int> selectedSet;
    selectedSet.reserve(selected.size());
    for (const auto& r : selected) {
        selectedSet.insert(r.label);
    }

    struct Acc {
        long long sx = 0;
        long long sy = 0;
        int count = 0;
    };

    std::unordered_map<int, Acc> acc;
    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            const int label = row[x];
            if (selectedSet.find(label) != selectedSet.end()) {
                auto& a = acc[label];
                a.sx += x;
                a.sy += y;
                a.count++;
            }
        }
    }

    std::unordered_map<int, cv::Point> centroids;
    for (const auto& kv : acc) {
        if (kv.second.count > 0) {
            centroids[kv.first] = cv::Point(static_cast<int>(kv.second.sx / kv.second.count),
                                            static_cast<int>(kv.second.sy / kv.second.count));
        }
    }
    return centroids;
}

struct QueueNode {
    HuffmanNode* node = nullptr;
    int minLeafLabel = -1;
    int serial = 0;
};

struct QueueNodeCmp {
    bool operator()(const QueueNode& a, const QueueNode& b) const {
        // 最小堆语义：权值小优先，其次叶子最小label小优先，最后serial小优先。
        if (a.node->weight != b.node->weight) {
            return a.node->weight > b.node->weight;
        }
        if (a.minLeafLabel != b.minLeafLabel) {
            return a.minLeafLabel > b.minLeafLabel;
        }
        return a.serial > b.serial;
    }
};

HuffmanNode* buildHuffmanTree(const std::vector<RegionArea>& selected) {
    if (selected.empty()) {
        return nullptr;
    }

    // 使用确定性 tie-break，保证相同输入下树结构稳定可复现。
    std::vector<RegionArea> leaves = selected;
    std::sort(leaves.begin(), leaves.end(), [](const RegionArea& a, const RegionArea& b) {
        if (a.area != b.area) {
            return a.area < b.area;
        }
        return a.label < b.label;
    });

    std::priority_queue<QueueNode, std::vector<QueueNode>, QueueNodeCmp> pq;
    int serial = 0;

    for (const auto& r : leaves) {
        auto* leaf = new HuffmanNode();
        leaf->weight = r.area;
        leaf->label = r.label;
        pq.push(QueueNode{leaf, r.label, serial++});
    }

    if (pq.size() == 1) {
        return pq.top().node;
    }

    while (pq.size() > 1) {
        QueueNode left = pq.top();
        pq.pop();
        QueueNode right = pq.top();
        pq.pop();

        auto* parent = new HuffmanNode();
        parent->weight = left.node->weight + right.node->weight;
        parent->label = -1;  // 非叶子保持空值语义。
        parent->left = left.node;
        parent->right = right.node;

        const int minLeafLabel = std::min(left.minLeafLabel, right.minLeafLabel);
        pq.push(QueueNode{parent, minLeafLabel, serial++});
    }

    return pq.top().node;
}

void generateHuffmanCodes(const HuffmanNode* root,
                          const std::string& prefix,
                          std::map<int, std::string>& outCodes,
                          int depth,
                          int& maxDepth) {
    if (!root) {
        return;
    }

    maxDepth = std::max(maxDepth, depth);
    if (!root->left && !root->right) {
        outCodes[root->label] = prefix.empty() ? "0" : prefix;
        return;
    }

    generateHuffmanCodes(root->left, prefix + "0", outCodes, depth + 1, maxDepth);
    generateHuffmanCodes(root->right, prefix + "1", outCodes, depth + 1, maxDepth);
}

int countLeaves(const HuffmanNode* root) {
    if (!root) {
        return 0;
    }
    if (!root->left && !root->right) {
        return 1;
    }
    return countLeaves(root->left) + countLeaves(root->right);
}

void drawHuffmanTree(const HuffmanNode* root,
                     cv::Mat& canvas,
                     int x,
                     int y,
                     int xOffset,
                     int levelHeight) {
    if (!root) {
        return;
    }

    const cv::Scalar edgeColor(90, 90, 90);
    const cv::Scalar nodeColor(255, 255, 255);
    const cv::Scalar textColor(30, 30, 30);

    if (root->left) {
        const int childX = x - xOffset;
        const int childY = y + levelHeight;
        cv::line(canvas, cv::Point(x, y), cv::Point(childX, childY), edgeColor, 1, cv::LINE_AA);
        cv::putText(canvas, "0", cv::Point((x + childX) / 2 - 8, (y + childY) / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
        drawHuffmanTree(root->left, canvas, childX, childY, std::max(24, xOffset / 2), levelHeight);
    }
    if (root->right) {
        const int childX = x + xOffset;
        const int childY = y + levelHeight;
        cv::line(canvas, cv::Point(x, y), cv::Point(childX, childY), edgeColor, 1, cv::LINE_AA);
        cv::putText(canvas, "1", cv::Point((x + childX) / 2 + 2, (y + childY) / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
        drawHuffmanTree(root->right, canvas, childX, childY, std::max(24, xOffset / 2), levelHeight);
    }

    cv::circle(canvas, cv::Point(x, y), 16, nodeColor, cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, cv::Point(x, y), 16, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);

    // 仅叶子节点显示值，非叶子节点保持空值。
    if (!root->left && !root->right) {
        const std::string txt = std::to_string(root->weight);
        cv::putText(canvas, txt, cv::Point(x - 14, y + 5), cv::FONT_HERSHEY_SIMPLEX,
                    0.35, textColor, 1, cv::LINE_AA);
    }
}

void destroyHuffmanTree(HuffmanNode* root) {
    if (!root) {
        return;
    }
    destroyHuffmanTree(root->left);
    destroyHuffmanTree(root->right);
    delete root;
}

}  // namespace task3

int main(int argc, char** argv) {
    const std::string inputPath = (argc >= 2) ? argv[1] : "../assets/fruits.jpg";
    const int k = (argc >= 3) ? std::max(10, std::atoi(argv[2])) : 120;

    cv::Mat src = cv::imread(inputPath);
    if (src.empty()) {
        std::cerr << "错误: 无法读取输入图像: " << inputPath << "\n";
        return 1;
    }

    std::vector<cv::Point> seeds = SeedSampler::generateSeeds(src.cols, src.rows, k);
    if (seeds.empty()) {
        std::cerr << "错误: 种子点生成失败\n";
        return 1;
    }

    cv::Mat markers = cv::Mat::zeros(src.size(), CV_32S);
    for (size_t i = 0; i < seeds.size(); ++i) {
        cv::circle(markers, seeds[i], 1, cv::Scalar(static_cast<int>(i + 1)), -1);
    }

    cv::Mat shifted;
    cv::pyrMeanShiftFiltering(src, shifted, 10, 51);
    cv::watershed(shifted, markers);

    std::vector<task3::RegionArea> areas = task3::buildRegionAreas(markers);
    if (areas.empty()) {
        std::cerr << "错误: 分水岭未得到有效区域\n";
        return 1;
    }

    task3::heapSortByAreaAscending(areas);

    const auto minArea = areas.front();
    const auto maxArea = areas.back();
    std::cout << "区域总数: " << areas.size() << "\n";
    std::cout << "最小面积: label=" << minArea.label << ", area=" << minArea.area << "\n";
    std::cout << "最大面积: label=" << maxArea.label << ", area=" << maxArea.area << "\n";

    std::cout << "请输入面积下界和上界(例如: 1000 5000): ";
    int low = 0;
    int high = 0;
    if (!(std::cin >> low >> high)) {
        std::cerr << "输入格式错误\n";
        return 1;
    }
    if (low > high) {
        std::swap(low, high);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    const auto range = task3::findAreaRangeBinarySearch(areas, low, high);
    if (range.first == -1) {
        std::cerr << "未找到面积落在该范围的区域\n";
        return 1;
    }

    std::vector<task3::RegionArea> selected;
    for (int i = range.first; i <= range.second; ++i) {
        selected.push_back(areas[i]);
    }

    std::cout << "匹配区域数: " << selected.size() << "\n";

    std::unordered_map<int, int> selectedArea;
    for (const auto& r : selected) {
        selectedArea[r.label] = r.area;
    }

    std::mt19937 rng(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> ch(80, 255);

    cv::Mat highlightMask = cv::Mat::zeros(src.size(), CV_8UC3);
    std::unordered_map<int, cv::Vec3b> colorByLabel;
    for (const auto& r : selected) {
        colorByLabel[r.label] = cv::Vec3b(static_cast<uchar>(ch(rng)),
                                          static_cast<uchar>(ch(rng)),
                                          static_cast<uchar>(ch(rng)));
    }

    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            const int label = markers.at<int>(y, x);
            if (label == -1) {
                highlightMask.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            } else if (selectedArea.find(label) != selectedArea.end()) {
                highlightMask.at<cv::Vec3b>(y, x) = colorByLabel[label];
            }
        }
    }

    cv::Mat highlighted;
    cv::addWeighted(src, 0.45, highlightMask, 0.55, 0.0, highlighted);

    const auto centroids = task3::computeRegionCentroids(markers, selected);
    for (const auto& r : selected) {
        auto it = centroids.find(r.label);
        if (it != centroids.end()) {
            const std::string text = std::to_string(r.area);
            cv::putText(highlighted, text, it->second, cv::FONT_HERSHEY_SIMPLEX,
                        0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }

    task3::HuffmanNode* root = task3::buildHuffmanTree(selected);
    if (!root) {
        std::cerr << "哈夫曼树构建失败\n";
        return 1;
    }

    std::map<int, std::string> codes;
    int maxDepth = 0;
    task3::generateHuffmanCodes(root, "", codes, 0, maxDepth);

    std::cout << "Huffman编码结果(label -> code):\n";
    for (const auto& r : selected) {
        auto it = codes.find(r.label);
        if (it != codes.end()) {
            std::cout << "  label=" << r.label << " (area=" << r.area << ") -> " << it->second << "\n";
        }
    }

    const int leaves = std::max(2, task3::countLeaves(root));
    const int width = std::max(1200, leaves * 80);
    const int height = std::max(600, (maxDepth + 2) * 120);
    cv::Mat treeCanvas(height, width, CV_8UC3, cv::Scalar(245, 245, 245));

    task3::drawHuffmanTree(root, treeCanvas, width / 2, 60, width / 4, 100);
    cv::putText(treeCanvas, "Task3 Huffman Tree", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(50, 50, 50), 2, cv::LINE_AA);

    fs::create_directories("../outputs");
    const std::string outHighlight = "../outputs/task3_highlight.png";
    const std::string outTree = "../outputs/task3_huffman_tree.png";

    cv::imwrite(outHighlight, highlighted);
    cv::imwrite(outTree, treeCanvas);

    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Task3 完成\n";
    std::cout << "高亮结果: " << outHighlight << "\n";
    std::cout << "哈夫曼树图: " << outTree << "\n";
    std::cout << "耗时: " << elapsedMs << " ms\n";

    task3::destroyHuffmanTree(root);
    return 0;
}
