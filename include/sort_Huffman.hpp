#ifndef SORT_HUFFMAN_HPP
#define SORT_HUFFMAN_HPP

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace task3 {

struct RegionArea {
    int label = 0;
    int area = 0;
};

struct HuffmanNode {
    int weight = 0;
    int label = -1;  // label >= 0 means leaf node.
    HuffmanNode* left = nullptr;
    HuffmanNode* right = nullptr;
};

std::vector<RegionArea> buildRegionAreas(const cv::Mat& markers);
void heapSortByAreaAscending(std::vector<RegionArea>& areas);
std::pair<int, int> findAreaRangeBinarySearch(const std::vector<RegionArea>& sortedAreasAsc,
                                              int low,
                                              int high);

std::unordered_map<int, cv::Point> computeRegionCentroids(const cv::Mat& markers,
                                                          const std::vector<RegionArea>& selected);

HuffmanNode* buildHuffmanTree(const std::vector<RegionArea>& selected);
void generateHuffmanCodes(const HuffmanNode* root,
                          const std::string& prefix,
                          std::map<int, std::string>& outCodes,
                          int depth,
                          int& maxDepth);
int countLeaves(const HuffmanNode* root);
void drawHuffmanTree(const HuffmanNode* root,
                     cv::Mat& canvas,
                     int x,
                     int y,
                     int xOffset,
                     int levelHeight);
void destroyHuffmanTree(HuffmanNode* root);

}  // namespace task3

#endif
