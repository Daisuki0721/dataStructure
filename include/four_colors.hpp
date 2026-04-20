#ifndef FOUR_COLOR_HPP
#define FOUR_COLOR_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fourcolor {

using AdjList = std::vector<std::vector<int>>;

struct ColoringStats {
    bool success = false;
    int backtracks = 0;
    int decisions = 0;
};

struct ColoringOptions {
    int maxBacktracks = 200000;
    double maxMs = 1500.0;
};

inline int bitCount4(uint8_t mask) {
    static const int lut[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
    return lut[mask & 0x0F];
}

inline std::vector<int> buildBfsOrder(const AdjList& adjacency) {
    const int n = static_cast<int>(adjacency.size());
    std::vector<int> order;
    order.reserve(n);

    if (n == 0) {
        return order;
    }

    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        degree[i] = static_cast<int>(adjacency[i].size());
    }

    std::vector<char> visited(n, 0);
    std::queue<int> q;

    auto pickStart = [&]() {
        int bestNode = -1;
        int bestDegree = -1;
        for (int i = 0; i < n; ++i) {
            if (!visited[i] && degree[i] > bestDegree) {
                bestDegree = degree[i];
                bestNode = i;
            }
        }
        return bestNode;
    };

    while (static_cast<int>(order.size()) < n) {
        const int start = pickStart();
        if (start == -1) {
            break;
        }

        visited[start] = 1;
        q.push(start);

        while (!q.empty()) {
            const int u = q.front();
            q.pop();
            order.push_back(u);

            std::vector<int> neighbors = adjacency[u];
            std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
                if (degree[a] != degree[b]) {
                    return degree[a] > degree[b];
                }
                return a < b;
            });

            for (int v : neighbors) {
                if (!visited[v]) {
                    visited[v] = 1;
                    q.push(v);
                }
            }
        }
    }

    return order;
}

inline AdjList buildAdjacencyFromWatershed(const cv::Mat& markers,
                                           std::vector<int>& regionLabels,
                                           std::unordered_map<int, int>& labelToNode) {
    std::set<int> uniqueLabels;
    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            if (row[x] > 0) {
                uniqueLabels.insert(row[x]);
            }
        }
    }

    regionLabels.assign(uniqueLabels.begin(), uniqueLabels.end());
    labelToNode.clear();
    for (int i = 0; i < static_cast<int>(regionLabels.size()); ++i) {
        labelToNode[regionLabels[i]] = i;
    }

    std::vector<std::unordered_set<int>> edgeSet(regionLabels.size());

    auto addEdgeByLabel = [&](int aLabel, int bLabel) {
        if (aLabel <= 0 || bLabel <= 0 || aLabel == bLabel) {
            return;
        }
        auto ita = labelToNode.find(aLabel);
        auto itb = labelToNode.find(bLabel);
        if (ita == labelToNode.end() || itb == labelToNode.end()) {
            return;
        }
        const int a = ita->second;
        const int b = itb->second;
        edgeSet[a].insert(b);
        edgeSet[b].insert(a);
    };

    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            const int curr = row[x];
            if (x + 1 < markers.cols) {
                addEdgeByLabel(curr, row[x + 1]);
            }
            if (y + 1 < markers.rows) {
                addEdgeByLabel(curr, markers.at<int>(y + 1, x));
            }
        }
    }

    // 对边界像素(-1)恢复区域邻接：使用保守的“对向关系”，避免过度加边。
    for (int y = 1; y < markers.rows - 1; ++y) {
        for (int x = 1; x < markers.cols - 1; ++x) {
            if (markers.at<int>(y, x) != -1) {
                continue;
            }

            const int up = markers.at<int>(y - 1, x);
            const int down = markers.at<int>(y + 1, x);
            const int left = markers.at<int>(y, x - 1);
            const int right = markers.at<int>(y, x + 1);

            addEdgeByLabel(up, down);
            addEdgeByLabel(left, right);

            std::vector<int> local;
            local.reserve(4);
            if (up > 0) local.push_back(up);
            if (down > 0) local.push_back(down);
            if (left > 0) local.push_back(left);
            if (right > 0) local.push_back(right);
            std::sort(local.begin(), local.end());
            local.erase(std::unique(local.begin(), local.end()), local.end());
            if (local.size() == 2) {
                addEdgeByLabel(local[0], local[1]);
            }
        }
    }

    AdjList adjacency(edgeSet.size());
    for (int i = 0; i < static_cast<int>(edgeSet.size()); ++i) {
        adjacency[i].assign(edgeSet[i].begin(), edgeSet[i].end());
        std::sort(adjacency[i].begin(), adjacency[i].end());
    }

    return adjacency;
}

inline std::array<int, 4> rankColors(int node,
                                     const AdjList& adjacency,
                                     const std::vector<int>& nodeColors,
                                     const std::array<int, 4>& colorUsage) {
    std::array<int, 4> localUse{0, 0, 0, 0};
    for (int nb : adjacency[node]) {
        const int c = nodeColors[nb];
        if (c >= 0 && c < 4) {
            localUse[c]++;
        }
    }

    std::array<int, 4> colors{0, 1, 2, 3};
    std::sort(colors.begin(), colors.end(), [&](int a, int b) {
        if (localUse[a] != localUse[b]) {
            return localUse[a] < localUse[b];
        }
        if (colorUsage[a] != colorUsage[b]) {
            return colorUsage[a] < colorUsage[b];
        }
        return a < b;
    });

    return colors;
}

inline bool validateColoring(const AdjList& adjacency, const std::vector<int>& nodeColors) {
    if (adjacency.size() != nodeColors.size()) {
        return false;
    }
    for (int u = 0; u < static_cast<int>(adjacency.size()); ++u) {
        if (nodeColors[u] < 0 || nodeColors[u] > 3) {
            return false;
        }
        for (int v : adjacency[u]) {
            if (v >= 0 && v < static_cast<int>(adjacency.size()) && nodeColors[u] == nodeColors[v]) {
                return false;
            }
        }
    }
    return true;
}

inline ColoringStats colorWithBfsStackBacktracking(const AdjList& adjacency,
                                                   const std::vector<int>& bfsOrder,
                                                   std::vector<int>& nodeColors,
                                                   const ColoringOptions& options = ColoringOptions()) {
    const int n = static_cast<int>(adjacency.size());
    nodeColors.assign(n, -1);

    ColoringStats stats;
    if (n == 0) {
        stats.success = true;
        return stats;
    }

    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        degree[i] = static_cast<int>(adjacency[i].size());
    }

    std::vector<int> bfsRank(n, n + 1);
    for (int i = 0; i < static_cast<int>(bfsOrder.size()); ++i) {
        bfsRank[bfsOrder[i]] = i;
    }

    std::vector<uint8_t> forbidMask(n, 0);
    std::array<int, 4> colorUsage{0, 0, 0, 0};

    struct Frame {
        int node = -1;
        int color = -1;
        uint8_t tried = 0;
        std::vector<std::pair<int, uint8_t>> changed;
        std::vector<std::pair<int, int>> forcedAssignments;
    };

    std::vector<Frame> stack(n);

    auto chooseNextNode = [&]() {
        bool hasHighDegreeCandidate = false;
        for (int i = 0; i < n; ++i) {
            if (nodeColors[i] == -1 && degree[i] > 3) {
                hasHighDegreeCandidate = true;
                break;
            }
        }

        int best = -1;
        int bestSat = -1;
        int bestDeg = -1;
        int bestBfs = n + 1;

        for (int i = 0; i < n; ++i) {
            if (nodeColors[i] != -1) {
                continue;
            }
            if (hasHighDegreeCandidate && degree[i] <= 3) {
                continue;
            }
            const int sat = bitCount4(forbidMask[i]);
            if (sat > bestSat ||
                (sat == bestSat && degree[i] > bestDeg) ||
                (sat == bestSat && degree[i] == bestDeg && bfsRank[i] < bestBfs)) {
                best = i;
                bestSat = sat;
                bestDeg = degree[i];
                bestBfs = bfsRank[i];
            }
        }

        return best;
    };

    auto hasDeadNode = [&]() {
        for (int i = 0; i < n; ++i) {
            if (nodeColors[i] == -1 && (forbidMask[i] & 0x0F) == 0x0F) {
                return true;
            }
        }
        return false;
    };

    auto firstColorFromMask = [&](uint8_t mask) {
        for (int c = 0; c < 4; ++c) {
            if (mask & static_cast<uint8_t>(1u << c)) {
                return c;
            }
        }
        return -1;
    };

    auto rollbackFrameAssignment = [&](Frame& fr) {
        for (auto it = fr.changed.rbegin(); it != fr.changed.rend(); ++it) {
            forbidMask[it->first] = it->second;
        }
        fr.changed.clear();

        for (auto it = fr.forcedAssignments.rbegin(); it != fr.forcedAssignments.rend(); ++it) {
            const int node = it->first;
            const int color = it->second;
            if (nodeColors[node] != -1 && color >= 0 && color < 4) {
                nodeColors[node] = -1;
                colorUsage[color]--;
            }
        }
        fr.forcedAssignments.clear();

        if (fr.node != -1 && fr.color != -1) {
            nodeColors[fr.node] = -1;
            colorUsage[fr.color]--;
        }
        fr.color = -1;
    };

    auto propagateForcedAssignments = [&](Frame& fr) {
        while (true) {
            int forcedNode = -1;
            uint8_t forcedMask = 0;
            int bestSat = -1;
            int bestDeg = -1;
            int bestBfs = n + 1;

            for (int i = 0; i < n; ++i) {
                if (nodeColors[i] != -1) {
                    continue;
                }

                const uint8_t avail = static_cast<uint8_t>((~forbidMask[i]) & 0x0F);
                const int availCount = bitCount4(avail);
                if (availCount == 0) {
                    return false;
                }
                if (availCount != 1) {
                    continue;
                }

                const int sat = bitCount4(forbidMask[i]);
                if (sat > bestSat ||
                    (sat == bestSat && degree[i] > bestDeg) ||
                    (sat == bestSat && degree[i] == bestDeg && bfsRank[i] < bestBfs)) {
                    forcedNode = i;
                    forcedMask = avail;
                    bestSat = sat;
                    bestDeg = degree[i];
                    bestBfs = bfsRank[i];
                }
            }

            if (forcedNode == -1) {
                return !hasDeadNode();
            }

            const int forcedColor = firstColorFromMask(forcedMask);
            if (forcedColor < 0) {
                return false;
            }

            const uint8_t bit = static_cast<uint8_t>(1u << forcedColor);
            nodeColors[forcedNode] = forcedColor;
            colorUsage[forcedColor]++;
            stats.decisions++;
            fr.forcedAssignments.push_back(std::make_pair(forcedNode, forcedColor));

            for (int nb : adjacency[forcedNode]) {
                if (nodeColors[nb] != -1) {
                    continue;
                }
                const uint8_t oldMask = forbidMask[nb];
                const uint8_t newMask = static_cast<uint8_t>(oldMask | bit);
                if (newMask != oldMask) {
                    fr.changed.push_back(std::make_pair(nb, oldMask));
                    forbidMask[nb] = newMask;
                }
            }

            if (hasDeadNode()) {
                return false;
            }
        }
    };

    const auto start = std::chrono::steady_clock::now();

    int depth = 0;
    while (depth >= 0) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(now - start).count();
        if (stats.backtracks > options.maxBacktracks || elapsedMs > options.maxMs) {
            stats.success = false;
            return stats;
        }

        if (depth == n) {
            stats.success = validateColoring(adjacency, nodeColors);
            return stats;
        }

        Frame& f = stack[depth];
        if (f.node == -1) {
            f.node = chooseNextNode();
            f.tried = 0;
            f.color = -1;
            f.changed.clear();
            f.forcedAssignments.clear();
            if (f.node == -1) {
                stats.success = validateColoring(adjacency, nodeColors);
                return stats;
            }
        }

        const uint8_t available = static_cast<uint8_t>((~forbidMask[f.node]) & 0x0F & (~f.tried));

        bool movedForward = false;
        if (available != 0) {
            const std::array<int, 4> candidates = rankColors(f.node, adjacency, nodeColors, colorUsage);
            for (int c : candidates) {
                const uint8_t bit = static_cast<uint8_t>(1u << c);
                if ((available & bit) == 0) {
                    continue;
                }

                f.tried = static_cast<uint8_t>(f.tried | bit);
                f.color = c;
                nodeColors[f.node] = c;
                colorUsage[c]++;
                stats.decisions++;

                f.changed.clear();
                f.forcedAssignments.clear();
                for (int nb : adjacency[f.node]) {
                    if (nodeColors[nb] != -1) {
                        continue;
                    }
                    const uint8_t oldMask = forbidMask[nb];
                    const uint8_t newMask = static_cast<uint8_t>(oldMask | bit);
                    if (newMask != oldMask) {
                        f.changed.push_back(std::make_pair(nb, oldMask));
                        forbidMask[nb] = newMask;
                    }
                }

                const bool propagationOk = !hasDeadNode() && propagateForcedAssignments(f);
                if (propagationOk) {
                    depth++;
                    if (depth < n) {
                        stack[depth].node = -1;
                        stack[depth].tried = 0;
                        stack[depth].color = -1;
                        stack[depth].changed.clear();
                        stack[depth].forcedAssignments.clear();
                    }
                    movedForward = true;
                    break;
                }

                rollbackFrameAssignment(f);
            }
        }

        if (movedForward) {
            continue;
        }

        f.node = -1;
        f.tried = 0;
        f.color = -1;
        f.changed.clear();
        f.forcedAssignments.clear();

        if (depth == 0) {
            break;
        }

        depth--;
        Frame& parent = stack[depth];
        if (parent.node != -1 && parent.color != -1) {
            rollbackFrameAssignment(parent);
            stats.backtracks++;
        }
    }

    stats.success = false;
    return stats;
}

}  // namespace fourcolor

#endif
