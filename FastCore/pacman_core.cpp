// Fast Pac-Man game, move features, tree evaluation and training games.
//
// A C++ port of PacmanGame/pacman.py (arcade level-1 rules),
// PacmanGame/board_graph.py (static graph and the 23 move features) and the
// game loop of GeneticAlgo/train_board_graph.evaluate_population_lockstep.
// Python keeps everything else (evolution, files, rendering). Built as the
// Python module FastCore._pacman_core by FastCore/build.sh.
//
// Same rules as the Python game; results are meant to match it closely, not
// bit for bit. Frightened ghosts use a port of Python's random.Random(0), so
// they turn as in Python; math.hypot is std::hypot, which can differ in the
// last bit.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// ---- Tile flags (PacmanGame/render_board.py) --------------------------------
constexpr int WALL = 1 << 0;
constexpr int PELLET = 1 << 1;
constexpr int POWER_PELLET = 1 << 2;
constexpr int GHOST_GATE = 1 << 3;
constexpr int PLAYER_LOCATION = 1 << 4;

// ---- Directions: ACTIONS order is up, down, left, right ---------------------
constexpr int UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3, STAY = -1;
constexpr int DX[4] = {0, 0, -1, 1};
constexpr int DY[4] = {-1, 1, 0, 0};
constexpr int REVERSE[4] = {DOWN, UP, RIGHT, LEFT};
constexpr int GHOST_ORDER[4] = {UP, LEFT, DOWN, RIGHT};  // GHOST_DIRECTION_ORDER
int ghost_order_index(int direction) {
    for (int i = 0; i < 4; ++i)
        if (GHOST_ORDER[i] == direction) return i;
    return 4;
}

// ---- Arcade constants (pacman.py) -------------------------------------------
constexpr double FRAMES_PER_SECOND = 60;
const double FULL_SPEED_TILES_PER_FRAME = 75.75757625 / FRAMES_PER_SECOND / 8;
constexpr double PACMAN_SPEED = 0.80, PACMAN_FRIGHT_SPEED = 0.90;
constexpr double DOT_PAUSE_FRAMES = 1, ENERGIZER_PAUSE_FRAMES = 3;
constexpr double GHOST_SPEED = 0.75, GHOST_FRIGHT_SPEED = 0.50, GHOST_TUNNEL_SPEED = 0.40;
constexpr double EYES_SPEED = 2.0;
constexpr double FRIGHT_FRAMES = 6 * FRAMES_PER_SECOND;
const double FRIGHT_TURN_FRAMES = 1.0 / (FULL_SPEED_TILES_PER_FRAME * PACMAN_FRIGHT_SPEED);
constexpr int SCATTER_CHASE_SECONDS[7] = {7, 20, 7, 20, 5, 20, 5};
constexpr double HOUSE_IDLE_FRAMES = 4 * FRAMES_PER_SECOND;
constexpr int PELLET_SCORE = 1000, POWER_PELLET_SCORE = 1000, GHOST_SCORE = 100;
// Ghost order everywhere is GHOST_KEYS: blinky, pinky, inky, clyde.
constexpr int BLINKY = 0, PINKY = 1, INKY = 2, CLYDE = 3;
constexpr int HOUSE_DOT_LIMITS[4] = {0, 0, 30, 60};
constexpr int GHOST_NEAR_RADIUS = 8;
constexpr int VISIT_WINDOW = 12;

// ---- Python's random.Random(0), enough for randrange(4) ---------------------
class PythonRandom {
  public:
    explicit PythonRandom(uint32_t seed_word) {
        // random.seed(int) -> init_by_array([abs(seed) as 32-bit words]).
        uint32_t key[1] = {seed_word};
        init_genrand(19650218U);
        int i = 1, j = 0;
        for (int k = 624; k; --k) {
            state_[i] = (state_[i] ^ ((state_[i - 1] ^ (state_[i - 1] >> 30)) * 1664525U)) + key[j] + j;
            ++i; ++j;
            if (i >= 624) { state_[0] = state_[623]; i = 1; }
            if (j >= 1) j = 0;
        }
        for (int k = 623; k; --k) {
            state_[i] = (state_[i] ^ ((state_[i - 1] ^ (state_[i - 1] >> 30)) * 1566083941U)) - i;
            ++i;
            if (i >= 624) { state_[0] = state_[623]; i = 1; }
        }
        state_[0] = 0x80000000U;
        index_ = 624;
    }
    // random.randrange(n) for small n: getrandbits(n.bit_length()) until < n.
    int randrange(int n) {
        int bits = 0;
        for (int v = n; v; v >>= 1) ++bits;
        while (true) {
            uint32_t r = next() >> (32 - bits);
            if (static_cast<int>(r) < n) return static_cast<int>(r);
        }
    }

  private:
    void init_genrand(uint32_t s) {
        state_[0] = s;
        for (int i = 1; i < 624; ++i)
            state_[i] = 1812433253U * (state_[i - 1] ^ (state_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        index_ = 624;
    }
    uint32_t next() {
        static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        if (index_ >= 624) {
            int kk = 0;
            for (; kk < 624 - 397; ++kk) {
                uint32_t y = (state_[kk] & 0x80000000U) | (state_[kk + 1] & 0x7fffffffU);
                state_[kk] = state_[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
            }
            for (; kk < 623; ++kk) {
                uint32_t y = (state_[kk] & 0x80000000U) | (state_[kk + 1] & 0x7fffffffU);
                state_[kk] = state_[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
            }
            uint32_t y = (state_[623] & 0x80000000U) | (state_[0] & 0x7fffffffU);
            state_[623] = state_[396] ^ (y >> 1) ^ mag01[y & 1U];
            index_ = 0;
        }
        uint32_t y = state_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
    uint32_t state_[624];
    int index_;
};

// ---- Static board graph (board_graph.StaticBoardGraph) ----------------------
struct Corridor {
    int end = -1;                 // point id of the last tile
    std::vector<int> tiles;       // cells, candidate first
    int exits = 0;                // corridor_exits
    int pocket_depth = 0;         // pocket_depths
    std::vector<std::pair<int, int>> continuations;  // (far junction id, length)
};

struct StaticGraph {
    int width = 0, height = 0, count = 0;
    std::vector<int> point_of_cell;   // cell -> point id, -1 if blocked
    std::vector<int> cell_of_point;
    std::vector<int> degree;          // by point id
    std::vector<int> neighbor;        // point id * 4 + action -> point id or -1
    std::vector<int32_t> distance;    // point id * count + point id
    int unreachable = 0;
    double scale = 1.0;
    std::vector<Corridor> corridors;  // point id * 4 + action

    int destination_cell(int cell, int action) const {
        int x = cell % width + DX[action], y = cell / width + DY[action];
        if (DY[action] == 0) x = ((x % width) + width) % width;
        if (x < 0 || x >= width || y < 0 || y >= height) return -1;
        return y * width + x;
    }
    int dist(int a_cell, int b_cell) const {
        int a = a_cell < 0 ? -1 : point_of_cell[a_cell];
        int b = b_cell < 0 ? -1 : point_of_cell[b_cell];
        if (a < 0 || b < 0) return unreachable;
        return distance[static_cast<size_t>(a) * count + b];
    }

    explicit StaticGraph(const std::vector<int>& tiles, int w, int h) : width(w), height(h) {
        point_of_cell.assign(w * h, -1);
        for (int cell = 0; cell < w * h; ++cell)
            if (!(tiles[cell] & (WALL | GHOST_GATE))) {
                point_of_cell[cell] = count++;
                cell_of_point.push_back(cell);
            }
        neighbor.assign(count * 4, -1);
        degree.assign(count, 0);
        for (int p = 0; p < count; ++p)
            for (int a = 0; a < 4; ++a) {
                int d = destination_cell(cell_of_point[p], a);
                if (d >= 0 && point_of_cell[d] >= 0) {
                    neighbor[p * 4 + a] = point_of_cell[d];
                    ++degree[p];
                }
            }
        unreachable = count + 1;
        distance.assign(static_cast<size_t>(count) * count, unreachable);
        std::vector<int> queue(count);
        int longest = 0;
        for (int s = 0; s < count; ++s) {
            int32_t* row = &distance[static_cast<size_t>(s) * count];
            int head = 0, tail = 0;
            row[s] = 0;
            queue[tail++] = s;
            while (head < tail) {
                int p = queue[head++];
                for (int a = 0; a < 4; ++a) {
                    int q = neighbor[p * 4 + a];
                    if (q >= 0 && row[q] == unreachable) {
                        row[q] = row[p] + 1;
                        longest = std::max(longest, row[q]);
                        queue[tail++] = q;
                    }
                }
            }
        }
        scale = static_cast<double>(longest + 1);
        build_corridors();
    }

    void build_corridors() {
        corridors.assign(count * 4, Corridor());
        for (int origin = 0; origin < count; ++origin)
            for (int a = 0; a < 4; ++a) {
                int candidate = neighbor[origin * 4 + a];
                if (candidate < 0) continue;
                Corridor& corridor = corridors[origin * 4 + a];
                std::vector<int> path = {candidate};
                int previous = origin, current = candidate;
                // The Python version loops forever on a ring with no junction;
                // stop after one lap here.
                while (degree[current] == 2 && static_cast<int>(path.size()) <= count) {
                    int onward = -1, onward_count = 0;
                    for (int b = 0; b < 4; ++b) {
                        int q = neighbor[current * 4 + b];
                        if (q >= 0 && q != previous) { onward = q; ++onward_count; }
                    }
                    if (onward_count != 1) break;
                    previous = current;
                    current = onward;
                    path.push_back(current);
                }
                corridor.end = path.back();
                for (int p : path) corridor.tiles.push_back(cell_of_point[p]);
                corridor.exits = std::max(0, degree[corridor.end] - 1);
            }
        for (int origin = 0; origin < count; ++origin)
            for (int a = 0; a < 4; ++a) {
                if (neighbor[origin * 4 + a] < 0) continue;
                Corridor& corridor = corridors[origin * 4 + a];
                int junction = corridor.end;
                int came_from = corridor.tiles.size() > 1
                    ? point_of_cell[corridor.tiles[corridor.tiles.size() - 2]] : origin;
                for (int b = 0; b < 4; ++b) {
                    int first = neighbor[junction * 4 + b];
                    if (first < 0 || first == came_from) continue;
                    const Corridor& onward = corridors[junction * 4 + b];
                    corridor.continuations.emplace_back(onward.end, static_cast<int>(onward.tiles.size()));
                }
                corridor.pocket_depth = pocket_depth(origin, neighbor[origin * 4 + a]);
            }
    }

    int pocket_depth(int origin, int start) const {
        int component = 0;
        const int32_t* row = &distance[static_cast<size_t>(origin) * count];
        for (int q = 0; q < count; ++q) component += row[q] < unreachable;
        std::vector<int> seen(count, -1);
        std::deque<int> queue = {start};
        seen[start] = 0;
        int seen_count = 1, deepest = 0;
        bool escaped = false;
        while (!queue.empty() && !escaped) {
            int tile = queue.front();
            queue.pop_front();
            for (int b = 0; b < 4; ++b) {
                int other = neighbor[tile * 4 + b];
                if (other < 0) continue;
                if (tile == start && other == origin) continue;  // the removed edge
                if (other == origin) { escaped = true; break; }
                if (seen[other] < 0) {
                    seen[other] = seen[tile] + 1;
                    deepest = std::max(deepest, seen[other]);
                    ++seen_count;
                    queue.push_back(other);
                }
            }
        }
        bool is_pocket = !escaped && 2 * seen_count < component;
        return is_pocket ? deepest + 1 : 0;
    }
};

// ---- Trees -----------------------------------------------------------------
enum Op : int32_t {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MIN, OP_MAX, OP_ABS, OP_NEG, OP_SQRT, OP_IF_LTE,
    OP_FEATURE, OP_CONSTANT
};
constexpr int FEATURES = 29;
// Tiles of each dangerous ghost's route that ghost_path_conflict predicts
// (board_graph.GHOST_PREDICTION_STEPS).
constexpr int GHOST_PREDICTION_STEPS = 10;
// Longer prediction for onward_safe_exits (board_graph.GHOST_FAR_PREDICTION_STEPS).
constexpr int GHOST_FAR_PREDICTION_STEPS = 25;
// Features from BASE_FEATURES on are the evasion features (approaching,
// pincer, safe_territory); they are only computed for trees that use them.
constexpr int BASE_FEATURES = 19;

struct Tree {
    std::vector<int32_t> ops;
    std::vector<double> values;  // feature index or constant value
    int evasion = 0;             // bit i set: reads evasion feature BASE_FEATURES + i
    std::vector<int32_t> end;    // index just past each node's subtree
    void finish() {
        // Subtree ends, from the back: each finished subtree pushes its end;
        // a node's end is the end of its last child.
        end.assign(ops.size(), 0);
        std::vector<int32_t> pending;
        for (size_t i = ops.size(); i-- > 0;) {
            int arity = ops[i] == OP_IF_LTE ? 4 : ops[i] >= OP_FEATURE ? 0 : ops[i] >= OP_ABS ? 1 : 2;
            int32_t e = static_cast<int32_t>(i) + 1;
            for (int c = 0; c < arity; ++c) { e = pending.back(); pending.pop_back(); }
            end[i] = e;
            pending.push_back(e);
        }
        evasion = 0;
        for (size_t i = 0; i < ops.size(); ++i)
            if (ops[i] == OP_FEATURE && values[i] >= BASE_FEATURES)
                evasion |= 1 << (static_cast<int>(values[i]) - BASE_FEATURES);
    }
};

// Python's min(a, b) and max(a, b): keep the first unless the second wins.
inline double py_min(double a, double b) { return b < a ? b : a; }
inline double py_max(double a, double b) { return b > a ? b : a; }

double evaluate(const Tree& tree, size_t& at, const double* features) {
    size_t node = at++;
    switch (tree.ops[node]) {
        case OP_ADD: { double a = evaluate(tree, at, features); double b = evaluate(tree, at, features); return a + b; }
        case OP_SUB: { double a = evaluate(tree, at, features); double b = evaluate(tree, at, features); return a - b; }
        case OP_MUL: { double a = evaluate(tree, at, features); double b = evaluate(tree, at, features); return a * b; }
        case OP_DIV: {  // gp_core.bounded_divide
            double a = evaluate(tree, at, features); double b = evaluate(tree, at, features);
            if (std::fabs(b) < 1e-3) b = b >= 0 ? 1e-3 : -1e-3;
            return py_max(-1000.0, py_min(1000.0, a / b));
        }
        case OP_MIN: { double a = evaluate(tree, at, features); double b = evaluate(tree, at, features); return py_min(a, b); }
        case OP_MAX: { double a = evaluate(tree, at, features); double b = evaluate(tree, at, features); return py_max(a, b); }
        case OP_ABS: return std::fabs(evaluate(tree, at, features));
        case OP_NEG: return -evaluate(tree, at, features);
        case OP_SQRT: return std::sqrt(std::fabs(evaluate(tree, at, features)));
        case OP_IF_LTE: {  // only the branch taken is evaluated (trees have no side effects)
            double a = evaluate(tree, at, features); double b = evaluate(tree, at, features);
            size_t otherwise_start = tree.end[at];
            double result;
            if (a <= b) {
                result = evaluate(tree, at, features);
                at = tree.end[otherwise_start];
            } else {
                at = otherwise_start;
                result = evaluate(tree, at, features);
            }
            return result;
        }
        case OP_FEATURE: return features[static_cast<int>(tree.values[node])];
        default: return tree.values[node];
    }
}

inline double safe_number(double value) {
    if (!std::isfinite(value)) return 0.0;
    return py_max(-1e9, py_min(1e9, value));
}

// ---- Starting position, as passed from Python --------------------------------
struct Start {
    int width = 0, height = 0;
    std::vector<int> tiles;
    int player = 0;              // cell
    int direction = LEFT;
    int ghosts[4] = {0, 0, 0, 0};  // cells, GHOST_KEYS order
    bool scared[4] = {false, false, false, false};
    int powerup_turns = 0;
    int pellets_eaten = 0;
    int pellets_remaining = 0;
    long long score = 0;
    bool original_maze = false;
    int powerup_duration = 52;
};

enum GhostState { HOUSE, LEAVING, ACTIVE };

// ---- One game (pacman.PacmanGame + board_graph.ObservationProvider) ---------
class Game {
  public:
    Game(const StaticGraph& graph, const Start& start) : g_(graph), s_(start), rng_(0) {
        width_ = start.width;
        height_ = start.height;
        tiles_ = start.tiles;
        player_ = start.player;
        direction_ = start.direction;
        score_ = start.score;
        pellets_eaten_ = start.pellets_eaten;
        pellets_remaining_ = start.pellets_remaining;
        powerup_turns_ = start.powerup_turns;
        for (int k = 0; k < 4; ++k) { ghost_[k] = start.ghosts[k]; home_[k] = start.ghosts[k]; }
        for (int cell = 0; cell < width_ * height_; ++cell)
            if (tiles_[cell] & GHOST_GATE) gates_.push_back(cell);
        find_house();
        // Observation provider state.
        is_pellet_.assign(width_ * height_, 0);
        is_power_.assign(width_ * height_, 0);
        pellet_slot_.assign(width_ * height_, -1);
        power_slot_.assign(width_ * height_, -1);
        for (int cell = 0; cell < width_ * height_; ++cell) {
            if (tiles_[cell] & PELLET) { is_pellet_[cell] = 1; add_target(pellet_ids_, pellet_slot_, cell); }
            if (tiles_[cell] & POWER_PELLET) { is_power_[cell] = 1; add_target(power_ids_, power_slot_, cell); }
            if (tiles_[cell] & (PELLET | POWER_PELLET)) {
                ++collectibles_;
                sum_x_ += cell % width_;
                sum_y_ += cell / width_;
            }
        }
        won_ = collectibles_ == 0;
        last_visit_.assign(width_ * height_, -1);
        last_visit_[player_] = 0;
        reset_dynamics();
    }

    bool over() const { return game_over_ || won_; }
    bool won() const { return won_; }
    bool game_over() const { return game_over_; }
    long long score() const { return score_; }
    int pellets_eaten() const { return pellets_eaten_; }
    int pellets_remaining() const { return pellets_remaining_; }
    int direction() const { return direction_; }

    // Features of every legal move, in ACTIONS order (for_legal_moves).
    // evasion: which evasion features to compute (bit i: feature BASE_FEATURES + i);
    // the others are set to 0.
    int observe(int moves[4], double features[4][FEATURES], int evasion = 1023) const {
        int count = 0;
        int origin = g_.point_of_cell[player_];
        if (origin < 0) return 0;
        int danger[4], edible[4], danger_k[4], danger_count = 0, edible_count = 0;
        for (int k = 0; k < 4; ++k) {
            if (frightened_[k]) edible[edible_count++] = ghost_[k];
            else { danger_k[danger_count] = k; danger[danger_count++] = ghost_[k]; }
        }
        if (evasion & (4 | 16 | 256)) ghost_distances(danger, danger_count);
        if (evasion & (32 | 512)) predict_ghost_paths();
        double elroy = (pellets_remaining_ <= 20 ? 0.5 : 0.0) + (pellets_remaining_ <= 10 ? 0.5 : 0.0);
        double frightened_time = std::min(1.0, static_cast<double>(powerup_turns_) / std::max(1, s_.powerup_duration));
        int pellet_total = pellets_eaten_ + pellets_remaining_;
        double pellets_left = pellet_total ? static_cast<double>(pellets_remaining_) / pellet_total : 0.0;
        for (int a = 0; a < 4; ++a) {
            int candidate_id = g_.neighbor[origin * 4 + a];
            if (candidate_id < 0) continue;
            int candidate = g_.cell_of_point[candidate_id];
            const Corridor& corridor = g_.corridors[origin * 4 + a];
            double* f = features[count];
            moves[count++] = a;
            int ghost_d[4], edible_d[4];
            for (int i = 0; i < danger_count; ++i) ghost_d[i] = g_.dist(candidate, danger[i]);
            std::sort(ghost_d, ghost_d + danger_count);
            for (int i = 0; i < edible_count; ++i) edible_d[i] = g_.dist(candidate, edible[i]);
            std::sort(edible_d, edible_d + edible_count);
            int nearby = 0;
            for (int i = 0; i < danger_count; ++i) nearby += ghost_d[i] <= GHOST_NEAR_RADIUS;
            int pellets_ahead = 0;
            for (int cell : corridor.tiles) pellets_ahead += is_pellet_[cell];
            double margin, safe;
            junction_safety(corridor, danger, danger_count, margin, safe);
            f[0] = is_pellet_[candidate];
            f[1] = normalized(nearest(candidate_id, pellet_ids_));
            f[2] = normalized(nearest(candidate_id, power_ids_));
            f[3] = normalized(danger_count > 0 ? ghost_d[0] : g_.unreachable);
            f[4] = normalized(danger_count > 1 ? ghost_d[1] : g_.unreachable);
            f[5] = static_cast<double>(nearby) / 4;
            f[6] = normalized(edible_count > 0 ? edible_d[0] : g_.unreachable);
            f[7] = frightened_time;
            f[8] = a == REVERSE[direction_];
            f[9] = a == direction_;
            f[10] = g_.degree[candidate_id] / 4.0;
            f[11] = static_cast<double>(pellets_ahead) / static_cast<double>(corridor.tiles.size());
            f[12] = std::min(1.0, static_cast<double>(corridor.tiles.size() - 1) / g_.scale);
            f[13] = corridor.exits / 3.0;
            f[14] = normalized(corridor.pocket_depth);
            f[15] = recently_visited(candidate);
            f[16] = pellet_center_alignment(a);
            f[17] = margin;
            f[18] = safe;
            f[19] = evasion & 1 ? danger_ghost_approaching(candidate_id, danger_k, danger_count) : 0.0;
            f[20] = evasion & 2 ? pincer(origin, candidate_id, danger_k, danger_count) : 0.0;
            f[21] = evasion & 4 ? safe_territory(candidate_id) : 0.0;
            f[22] = pellets_left;
            f[23] = evasion & 16 ? safe_pellet_distance(candidate_id) : 0.0;
            f[24] = evasion & 32 ? ghost_path_conflict(corridor) : 0.0;
            f[25] = normalized(frightened_[BLINKY] ? g_.unreachable : g_.dist(candidate, ghost_[BLINKY]));
            f[26] = elroy;
            f[27] = evasion & 256 ? safe_power_pellet_distance(candidate_id) : 0.0;
            f[28] = evasion & 512 ? onward_safe_exits(origin, corridor) : 0.0;
        }
        return count;
    }

    void step(int action) {
        if (over()) return;
        move_player(action);
        ++visit_clock_;
        last_visit_[player_] = visit_clock_;
        if (!game_over_) {
            advance_time();
            move_ghosts();
        }
        won_ = collectibles_ == 0;
    }

  private:
    double normalized(int d) const { return d >= g_.unreachable ? 1.0 : d / g_.scale; }

    // -- evasion features (board_graph.ObservationProvider._approaching etc.) --
    int point_dist(int a, int b) const { return g_.distance[static_cast<size_t>(a) * g_.count + b]; }

    // 1 if ghost k's next step can bring it closer to point target, 0 if not,
    // -1 if unknown (off the graph or unreachable).
    int approaching(int k, int target) const {
        int ghost = g_.point_of_cell[ghost_[k]];
        if (ghost < 0) return -1;
        int d = point_dist(ghost, target);
        if (d >= g_.unreachable) return -1;
        if (d == 0) return 1;
        int heading = ghost_direction_[k];
        int back = heading >= 0 ? REVERSE[heading] : -1;
        int best = std::numeric_limits<int>::max();
        bool any = false;
        for (int a = 0; a < 4; ++a) {
            int n = g_.neighbor[ghost * 4 + a];
            if (n < 0 || a == back) continue;
            any = true;
            best = std::min(best, point_dist(n, target));
        }
        if (!any)
            for (int a = 0; a < 4; ++a) {
                int n = g_.neighbor[ghost * 4 + a];
                if (n < 0) continue;
                any = true;
                best = std::min(best, point_dist(n, target));
            }
        if (!any) return -1;
        return best < d ? 1 : 0;
    }

    double danger_ghost_approaching(int candidate, const int* keys, int n) const {
        int nearest = -1, nearest_d = g_.unreachable;
        for (int i = 0; i < n; ++i) {
            int d = g_.dist(g_.cell_of_point[candidate], ghost_[keys[i]]);
            if (d < nearest_d) { nearest = keys[i]; nearest_d = d; }
        }
        if (nearest < 0) return 0.0;
        int a = approaching(nearest, candidate);
        return a < 0 ? 0.0 : a ? 1.0 : -1.0;
    }

    double pincer(int origin, int candidate, const int* keys, int n) const {
        bool ahead = false, behind = false;
        for (int i = 0; i < n; ++i) {
            int k = keys[i];
            int to_candidate = g_.dist(g_.cell_of_point[candidate], ghost_[k]);
            if (to_candidate > GHOST_NEAR_RADIUS) continue;
            int to_origin = g_.dist(g_.cell_of_point[origin], ghost_[k]);
            if (to_candidate < to_origin) ahead = true;
            else if (to_candidate > to_origin && approaching(k, origin) == 1) behind = true;
        }
        return ahead && behind ? 1.0 : 0.0;
    }

    // Nearest dangerous ghost's path distance to every point (distance_map).
    void ghost_distances(const int* ghosts, int n) const {
        gmin_.assign(g_.count, g_.unreachable);
        for (int i = 0; i < n; ++i) {
            int id = g_.point_of_cell[ghosts[i]];
            if (id < 0) continue;
            const int32_t* row = &g_.distance[static_cast<size_t>(id) * g_.count];
            for (int p = 0; p < g_.count; ++p) gmin_[p] = std::min(gmin_[p], static_cast<int>(row[p]));
        }
    }

    // Tiles with 1 + d(candidate, x) < nearest ghost's d(x): the same set as a
    // search through tiles Pac-Man reaches first (see board_graph.py).
    double safe_territory(int candidate) const {
        const int32_t* row = &g_.distance[static_cast<size_t>(candidate) * g_.count];
        const int* ghost = gmin_.data();
        int total = 0;
        for (int p = 0; p < g_.count; ++p) total += row[p] + 1 < ghost[p];
        return static_cast<double>(total) / g_.count;
    }

    // Nearest pellet or power pellet Pac-Man reaches strictly before every
    // dangerous ghost (board_graph.ObservationProvider._safe_pellet_distance).
    double safe_pellet_distance(int candidate) const {
        const int32_t* row = &g_.distance[static_cast<size_t>(candidate) * g_.count];
        int best = g_.unreachable;
        for (const std::vector<int>* ids : {&pellet_ids_, &power_ids_})
            for (int id : *ids)
                if (row[id] < best && row[id] + 1 < gmin_[id]) best = row[id];
        return normalized(best);
    }

    // Nearest power pellet Pac-Man reaches strictly before every dangerous
    // ghost (board_graph.ObservationProvider._safe_power_pellet_distance).
    double safe_power_pellet_distance(int candidate) const {
        const int32_t* row = &g_.distance[static_cast<size_t>(candidate) * g_.count];
        int best = g_.unreachable;
        for (int id : power_ids_)
            if (row[id] < best && row[id] + 1 < gmin_[id]) best = row[id];
        return normalized(best);
    }

    // Pac-Man on tiles[i] at time first_time + i meets a predicted ghost
    // (board_graph.ObservationProvider._route_meets).
    bool route_meets(const std::vector<int>& tiles, int first_time) const {
        for (int k = 0; k < 4; ++k)
            for (int t = 0; t < path_length_[k]; ++t)
                for (size_t i = 0; i < tiles.size(); ++i)
                    if (tiles[i] == path_[k][t] && std::abs(t - (first_time + static_cast<int>(i))) <= 1)
                        return true;
        return false;
    }

    // Share of the corridors beyond this move's far junction that stay clear
    // of the predicted routes (board_graph.ObservationProvider._onward_safe_exits).
    double onward_safe_exits(int origin, const Corridor& corridor) const {
        if (route_meets(corridor.tiles, 1)) return 0.0;
        int end = corridor.end;
        int came_from = corridor.tiles.size() > 1
            ? g_.point_of_cell[corridor.tiles[corridor.tiles.size() - 2]] : origin;
        int start = static_cast<int>(corridor.tiles.size()) + 1;
        int total = 0, clear = 0;
        for (int b = 0; b < 4; ++b) {
            int next = g_.neighbor[end * 4 + b];
            if (next < 0 || next == came_from) continue;
            ++total;
            clear += !route_meets(g_.corridors[end * 4 + b].tiles, start);
        }
        return total ? static_cast<double>(clear) / total : 0.0;
    }

    // Each active, unfrightened ghost's next GHOST_FAR_PREDICTION_STEPS tiles under
    // its normal turning rule, with Pac-Man and Blinky held where they are
    // (PacmanGame.predicted_ghost_paths). path_[k][0] is the current tile.
    void predict_ghost_paths() const {
        for (int k = 0; k < 4; ++k) {
            path_length_[k] = 0;
            if (state_[k] != ACTIVE || frightened_[k]) continue;
            int position = ghost_[k], previous = ghost_direction_[k];
            path_[k][path_length_[k]++] = position;
            for (int step = 0; step < GHOST_FAR_PREDICTION_STEPS; ++step) {
                int option[4] = {-1, -1, -1, -1};
                int option_count = 0;
                for (int direction : GHOST_ORDER) {
                    int next = destination(position, direction);
                    if (!can_enter(next, true)) continue;
                    if ((tiles_[next] & GHOST_GATE) || in_house_[next]) continue;
                    if (direction == UP && no_up_[position]) continue;
                    option[direction] = next;
                    ++option_count;
                }
                if (!option_count) break;
                if (previous >= 0 && option_count > 1 && option[REVERSE[previous]] >= 0) {
                    option[REVERSE[previous]] = -1;
                    --option_count;
                }
                long long tx, ty;
                ghost_target(k, position, tx, ty);
                long long best = std::numeric_limits<long long>::max();
                int best_order = 5, chosen = -1;
                for (int d = 0; d < 4; ++d) {
                    if (option[d] < 0) continue;
                    long long dist = distance_squared(cell_x(option[d]), cell_y(option[d]), tx, ty);
                    int order = ghost_order_index(d);
                    if (dist < best || (dist == best && order < best_order)) {
                        best = dist; best_order = order; chosen = d;
                    }
                }
                previous = chosen;
                position = option[chosen];
                path_[k][path_length_[k]++] = position;
            }
        }
    }

    // How soon a predicted ghost route meets Pac-Man on the move's corridor
    // (board_graph.ObservationProvider._ghost_path_conflict).
    double ghost_path_conflict(const Corridor& corridor) const {
        const int steps = GHOST_PREDICTION_STEPS;
        int tiles = std::min(static_cast<int>(corridor.tiles.size()), steps);
        int earliest = -1;
        for (int k = 0; k < 4; ++k)
            for (int t = 0; t < std::min(path_length_[k], steps + 1); ++t)
                for (int j = 0; j < tiles; ++j)
                    if (corridor.tiles[j] == path_[k][t] && std::abs(t - (j + 1)) <= 1) {
                        int meet = std::min(t, j + 1);
                        if (earliest < 0 || meet < earliest) earliest = meet;
                    }
        if (earliest < 0) return 0.0;
        return static_cast<double>(steps + 2 - std::max(1, earliest)) / (steps + 1);
    }

    // Point ids of the (power) pellets still on the board that lie on the
    // graph, kept as an unordered list; slots maps a cell to its position.
    void add_target(std::vector<int>& ids, std::vector<int>& slots, int cell) {
        int id = g_.point_of_cell[cell];
        if (id < 0) return;
        slots[cell] = static_cast<int>(ids.size());
        ids.push_back(id);
    }
    void remove_target(std::vector<int>& ids, std::vector<int>& slots, int cell) {
        int slot = slots[cell];
        if (slot < 0) return;
        int moved = ids.back();
        ids[slot] = moved;
        slots[g_.cell_of_point[moved]] = slot;
        slots[cell] = -1;
        ids.pop_back();
    }

    int nearest(int from_id, const std::vector<int>& ids) const {
        if (ids.empty()) return g_.unreachable;
        const int32_t* row = &g_.distance[static_cast<size_t>(from_id) * g_.count];
        int best = std::numeric_limits<int>::max();
        for (int id : ids) best = std::min(best, static_cast<int>(row[id]));
        return best;
    }

    double recently_visited(int cell) const {
        int last = last_visit_[cell];
        if (last < 0) return 0.0;
        int age = visit_clock_ - last;
        if (age > VISIT_WINDOW) return 0.0;
        return static_cast<double>(VISIT_WINDOW - std::max(age, 1) + 1) / VISIT_WINDOW;
    }

    double pellet_center_alignment(int action) const {
        if (!collectibles_) return 0.0;
        double dx = static_cast<double>(sum_x_) / collectibles_ - player_ % width_;
        double dy = static_cast<double>(sum_y_) / collectibles_ - player_ / width_;
        double magnitude = std::hypot(dx, dy);
        if (magnitude == 0.0) return 0.0;
        return (DX[action] * dx + DY[action] * dy) / magnitude;
    }

    void junction_safety(const Corridor& corridor, const int* ghosts, int ghost_count,
                         double& margin_out, double& safe_out) const {
        const int32_t* rows[4];
        int row_count = 0;
        for (int i = 0; i < ghost_count; ++i) {
            int id = g_.point_of_cell[ghosts[i]];
            if (id >= 0) rows[row_count++] = &g_.distance[static_cast<size_t>(id) * g_.count];
        }
        if (!row_count) {
            margin_out = 1.0;
            safe_out = corridor.continuations.size() / 3.0;
            return;
        }
        int steps = static_cast<int>(corridor.tiles.size());
        int nearest_ghost = std::numeric_limits<int>::max();
        for (int r = 0; r < row_count; ++r) nearest_ghost = std::min(nearest_ghost, static_cast<int>(rows[r][corridor.end]));
        int margin = nearest_ghost - steps;
        int safe = 0;
        if (margin > 0)
            for (const auto& [far, length] : corridor.continuations) {
                int closest = std::numeric_limits<int>::max();
                for (int r = 0; r < row_count; ++r) closest = std::min(closest, static_cast<int>(rows[r][far]));
                safe += closest > steps + length;
            }
        margin_out = std::max(-1.0, std::min(1.0, margin / g_.scale));
        safe_out = safe / 3.0;
    }

    // -- game rules --
    int cell_x(int cell) const { return cell % width_; }
    int cell_y(int cell) const { return cell / width_; }
    static long long distance_squared(long long ax, long long ay, long long bx, long long by) {
        return (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
    }
    int destination(int cell, int action, bool wrap = true) const {
        int x = cell_x(cell) + DX[action], y = cell_y(cell) + DY[action];
        if (wrap && DY[action] == 0) x = ((x % width_) + width_) % width_;
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return -1;
        return y * width_ + x;
    }
    bool can_enter(int cell, bool ghost) const {
        if (cell < 0) return false;
        return !(tiles_[cell] & (ghost ? WALL : (WALL | GHOST_GATE)));
    }

    void find_house() {
        in_house_.assign(width_ * height_, 0);
        if (gates_.empty()) return;
        int gate = gates_[gates_.size() / 2];
        int seed = cell_y(gate) + 1 < height_ ? gate + width_ : -1;
        if (seed < 0 || tiles_[seed] & (WALL | GHOST_GATE)) return;
        std::deque<int> pending = {seed};
        in_house_[seed] = 1;
        house_cells_.push_back(seed);
        const int order[4] = {UP, LEFT, RIGHT, DOWN};
        while (!pending.empty()) {
            int cell = pending.front();
            pending.pop_front();
            for (int a : order) {
                int next = destination(cell, a, false);
                if (next >= 0 && !in_house_[next] && !(tiles_[next] & (WALL | GHOST_GATE))) {
                    in_house_[next] = 1;
                    house_cells_.push_back(next);
                    pending.push_back(next);
                }
            }
        }
    }

    void reset_dynamics() {
        for (int k = 0; k < 4; ++k) {
            ghost_direction_[k] = -1;
            if (in_house_[ghost_[k]]) state_[k] = HOUSE;
            else if (tiles_[ghost_[k]] & GHOST_GATE) state_[k] = LEAVING;
            else state_[k] = ACTIVE;
            progress_[k] = 0.0;
            reverse_[k] = false;
            respawn_[k] = 0.0;
            house_dots_[k] = pellets_eaten_;
        }
        fright_frames_ = powerup_turns_ * FRIGHT_TURN_FRAMES;
        for (int k = 0; k < 4; ++k) frightened_[k] = fright_frames_ > 0 && s_.scared[k];
        tunnel_.assign(width_ * height_, 0);
        for (int y = 0; y < height_; ++y) {
            int left = y * width_, right = y * width_ + width_ - 1;
            if (g_.point_of_cell[left] < 0 || g_.point_of_cell[right] < 0) continue;
            for (int x = 0; x < width_; ++x) {
                int id = g_.point_of_cell[y * width_ + x];
                if (id < 0 || g_.degree[id] != 2) break;
                tunnel_[y * width_ + x] = 1;
            }
            for (int x = width_ - 1; x >= 0; --x) {
                int id = g_.point_of_cell[y * width_ + x];
                if (id < 0 || g_.degree[id] != 2) break;
                tunnel_[y * width_ + x] = 1;
            }
        }
        no_up_.assign(width_ * height_, 0);
        if (s_.original_maze) {
            const int tiles[4][2] = {{12, 11}, {15, 11}, {12, 23}, {15, 23}};
            for (const auto& t : tiles)
                if (t[0] < width_ && t[1] < height_) no_up_[t[1] * width_ + t[0]] = 1;
        }
        exit_.assign(width_ * height_, 0);
        for (int gate : gates_)
            if (cell_y(gate) > 0) { exit_[gate - width_] = 1; exit_cells_.push_back(gate - width_); }
        uses_marker_ = false;
        for (int value : tiles_) uses_marker_ |= (value & PLAYER_LOCATION) != 0;
        release_ghosts();
    }

    bool scatter_mode() const { return mode_index_ < 7 && mode_index_ % 2 == 0; }

    void release_ghosts() {
        for (int k = 0; k < 4; ++k) {
            if (state_[k] != HOUSE || respawn_[k] > 0) continue;
            if (house_dots_[k] < HOUSE_DOT_LIMITS[k]) break;
            start_leaving(k);
        }
    }
    void start_leaving(int k) {
        state_[k] = in_house_[ghost_[k]] ? LEAVING : ACTIVE;
        progress_[k] = 0.0;
    }
    int preferred_waiting_ghost() const {
        for (int k = 0; k < 4; ++k)
            if (state_[k] == HOUSE && respawn_[k] <= 0) return k;
        return -1;
    }
    void frighten_ghosts() {
        fright_frames_ = FRIGHT_FRAMES;
        for (int k = 0; k < 4; ++k) {
            if (respawn_[k] > 0) continue;
            frightened_[k] = true;
            if (state_[k] == ACTIVE) reverse_[k] = true;
        }
    }

    void move_player(int action) {
        if (action != STAY) direction_ = action;
        int target = action == STAY ? player_ : destination(player_, action);
        if (!can_enter(target, false)) target = player_;
        bool was_frightened = fright_frames_ > 0;
        player_ = target;
        if (resolve_collisions()) return;
        int tile = tiles_[target];
        if (tile & PELLET) score_ += PELLET_SCORE;
        if (tile & POWER_PELLET) score_ += POWER_PELLET_SCORE;
        int consumed = (tile & (PELLET | POWER_PELLET)) ? 1 : 0;
        tiles_[target] = tile & ~(PELLET | POWER_PELLET);
        if (consumed) {
            if (is_pellet_[target]) remove_target(pellet_ids_, pellet_slot_, target);
            if (is_power_[target]) remove_target(power_ids_, power_slot_, target);
            is_pellet_[target] = 0;
            is_power_[target] = 0;
            --collectibles_;
            sum_x_ -= cell_x(target);
            sum_y_ -= cell_y(target);
        }
        pellets_eaten_ += consumed;
        pellets_remaining_ = std::max(0, pellets_remaining_ - consumed);
        double speed = was_frightened ? PACMAN_FRIGHT_SPEED : PACMAN_SPEED;
        step_frames_ = 1.0 / (FULL_SPEED_TILES_PER_FRAME * speed);
        ate_dot_ = consumed != 0;
        if (tile & POWER_PELLET) {
            step_frames_ += ENERGIZER_PAUSE_FRAMES;
            frighten_ghosts();
        } else if (tile & PELLET) {
            step_frames_ += DOT_PAUSE_FRAMES;
        }
        // The bit-16 marker does not affect play; it is not tracked here.
    }

    bool resolve_collisions() {
        for (int k = 0; k < 4; ++k) {
            if (state_[k] == HOUSE || ghost_[k] != player_) continue;
            if (frightened_[k]) {
                send_ghost_home(k);
            } else {
                game_over_ = true;
                return true;
            }
        }
        return false;
    }

    void send_ghost_home(int k) {
        int eaten_at = ghost_[k];
        score_ += GHOST_SCORE;
        ghost_direction_[k] = -1;
        frightened_[k] = false;
        reverse_[k] = false;
        progress_[k] = 0.0;
        state_[k] = HOUSE;
        int home = home_[k];
        if (!house_cells_.empty() && !in_house_[home]) {
            long long best_d = std::numeric_limits<long long>::max();
            int best_y = 0, best_x = 0;
            home = -1;
            for (int cell : house_cells_) {
                long long d = std::numeric_limits<long long>::max();
                for (int gate : gates_)
                    d = std::min(d, distance_squared(cell_x(cell), cell_y(cell), cell_x(gate), cell_y(gate)));
                int x = cell_x(cell), y = cell_y(cell);
                if (home < 0 || std::make_tuple(d, y, x) < std::make_tuple(best_d, best_y, best_x)) {
                    home = cell; best_d = d; best_y = y; best_x = x;
                }
            }
        }
        int distance;
        if (!exit_cells_.empty()) {
            distance = std::numeric_limits<int>::max();
            for (int cell : exit_cells_) distance = std::min(distance, g_.dist(eaten_at, cell));
        } else {
            distance = g_.dist(eaten_at, home);
        }
        if (distance >= g_.unreachable) distance = 0;
        respawn_[k] = std::max(1.0, distance / (FULL_SPEED_TILES_PER_FRAME * EYES_SPEED));
        ghost_[k] = home;
    }

    void ahead_of_player(int distance, long long& x, long long& y) const {
        x = cell_x(player_) + DX[direction_] * distance;
        y = cell_y(player_) + DY[direction_] * distance;
        if (direction_ == UP) x -= distance;
    }
    void scatter_target(int k, long long& x, long long& y) const {
        switch (k) {
            case BLINKY: x = width_ - 3; y = -3; break;
            case PINKY: x = 2; y = -3; break;
            case INKY: x = width_ - 1; y = height_ + 1; break;
            default: x = 0; y = height_ + 1; break;
        }
    }
    double elroy_speed() const {  // 0 when not in a Cruise Elroy stage
        if (pellets_remaining_ <= 10) return 0.85;
        if (pellets_remaining_ <= 20) return 0.80;
        return 0.0;
    }
    void ghost_target(int k, int position, long long& x, long long& y) const {
        if (state_[k] == LEAVING) {
            int gate = position;
            long long best = std::numeric_limits<long long>::max();
            for (int candidate : gates_) {
                long long d = distance_squared(cell_x(position), cell_y(position), cell_x(candidate), cell_y(candidate));
                if (d < best) { best = d; gate = candidate; }
            }
            x = cell_x(gate);
            y = cell_y(gate) - 1;
            return;
        }
        bool elroy = k == BLINKY && elroy_speed() > 0;
        if (scatter_mode() && !elroy) { scatter_target(k, x, y); return; }
        if (k == BLINKY) { x = cell_x(player_); y = cell_y(player_); return; }
        if (k == PINKY) { ahead_of_player(4, x, y); return; }
        if (k == INKY) {
            long long ax, ay;
            ahead_of_player(2, ax, ay);
            x = ax * 2 - cell_x(ghost_[BLINKY]);
            y = ay * 2 - cell_y(ghost_[BLINKY]);
            return;
        }
        if (distance_squared(cell_x(position), cell_y(position), cell_x(player_), cell_y(player_)) >= 64) {
            x = cell_x(player_); y = cell_y(player_); return;
        }
        scatter_target(k, x, y);
    }
    double ghost_speed(int k) const {
        if (tunnel_[ghost_[k]]) return GHOST_TUNNEL_SPEED;
        if (frightened_[k]) return GHOST_FRIGHT_SPEED;
        if (k == BLINKY) { double e = elroy_speed(); return e > 0 ? e : GHOST_SPEED; }
        return GHOST_SPEED;
    }

    void step_ghost(int k) {
        int position = ghost_[k];
        bool leaving = state_[k] == LEAVING;
        bool frightened = frightened_[k] && !leaving;
        int option[4] = {-1, -1, -1, -1};  // destination by direction
        int option_count = 0;
        for (int direction : GHOST_ORDER) {
            int next = destination(position, direction);
            if (!can_enter(next, true)) continue;
            if (!leaving && ((tiles_[next] & GHOST_GATE) || in_house_[next])) continue;
            if (direction == UP && !frightened && !leaving && no_up_[position]) continue;
            option[direction] = next;
            ++option_count;
        }
        if (!option_count) return;
        int previous = ghost_direction_[k];
        int direction = -1;
        if (reverse_[k]) {
            reverse_[k] = false;
            if (previous >= 0 && option[REVERSE[previous]] >= 0) direction = REVERSE[previous];
        }
        if (direction < 0) {
            if (previous >= 0 && option_count > 1 && option[REVERSE[previous]] >= 0) {
                option[REVERSE[previous]] = -1;
                --option_count;
            }
            if (frightened) {
                int first = rng_.randrange(4);
                for (int i = 0; i < 4; ++i) {
                    int d = GHOST_ORDER[(first + i) % 4];
                    if (option[d] >= 0) { direction = d; break; }
                }
            } else {
                long long tx, ty;
                ghost_target(k, position, tx, ty);
                long long best = std::numeric_limits<long long>::max();
                int best_order = 5;
                for (int d = 0; d < 4; ++d) {
                    if (option[d] < 0) continue;
                    long long dist = distance_squared(cell_x(option[d]), cell_y(option[d]), tx, ty);
                    int order = ghost_order_index(d);
                    if (dist < best || (dist == best && order < best_order)) {
                        best = dist; best_order = order; direction = d;
                    }
                }
            }
        }
        int next = option[direction];
        ghost_direction_[k] = direction;
        ghost_[k] = next;
        if (leaving && exit_[next]) {
            state_[k] = ACTIVE;
            ghost_direction_[k] = LEFT;
        }
    }

    void advance_time() {
        double frames = step_frames_;
        for (int k = 0; k < 4; ++k)
            if (respawn_[k] > 0) respawn_[k] = std::max(0.0, respawn_[k] - frames);
        if (fright_frames_ > 0) {
            if (fright_frames_ < FRIGHT_FRAMES || !ate_dot_) fright_frames_ = std::max(0.0, fright_frames_ - frames);
            if (fright_frames_ == 0)
                for (int k = 0; k < 4; ++k) frightened_[k] = false;
        } else {
            mode_frames_ += frames;
            while (mode_index_ < 7 && mode_frames_ >= mode_boundary(mode_index_)) {
                ++mode_index_;
                for (int k = 0; k < 4; ++k)
                    if (state_[k] == ACTIVE) reverse_[k] = true;
            }
        }
        if (ate_dot_) {
            dot_idle_frames_ = 0.0;
            int waiting = preferred_waiting_ghost();
            if (waiting >= 0) ++house_dots_[waiting];
        } else {
            dot_idle_frames_ += frames;
            if (dot_idle_frames_ >= HOUSE_IDLE_FRAMES) {
                dot_idle_frames_ = 0.0;
                int waiting = preferred_waiting_ghost();
                if (waiting >= 0) start_leaving(waiting);
            }
        }
        release_ghosts();
        powerup_turns_ = static_cast<int>(std::ceil(fright_frames_ / FRIGHT_TURN_FRAMES));
    }
    static double mode_boundary(int index) {
        int seconds = 0;
        for (int i = 0; i <= index; ++i) seconds += SCATTER_CHASE_SECONDS[i];
        return seconds * FRAMES_PER_SECOND;
    }

    void move_ghosts() {
        for (int k = 0; k < 4; ++k) {
            if (state_[k] == HOUSE) continue;
            progress_[k] += step_frames_ * FULL_SPEED_TILES_PER_FRAME * ghost_speed(k);
            while (progress_[k] >= 1.0) {
                progress_[k] -= 1.0;
                step_ghost(k);
                if (resolve_collisions()) return;
                if (state_[k] == HOUSE) break;
            }
        }
    }

    const StaticGraph& g_;
    const Start& s_;
    PythonRandom rng_;
    int width_ = 0, height_ = 0;
    std::vector<int> tiles_;
    int player_ = 0, direction_ = LEFT;
    long long score_ = 0;
    int pellets_eaten_ = 0, pellets_remaining_ = 0, powerup_turns_ = 0;
    bool game_over_ = false, won_ = false;
    int ghost_[4], home_[4], ghost_direction_[4];
    GhostState state_[4];
    double progress_[4], respawn_[4];
    bool reverse_[4], frightened_[4];
    int house_dots_[4];
    double dot_idle_frames_ = 0, mode_frames_ = 0, fright_frames_ = 0, step_frames_ = 0;
    int mode_index_ = 0;
    bool ate_dot_ = false, uses_marker_ = false;
    std::vector<int> gates_, house_cells_, exit_cells_;
    std::vector<char> in_house_, tunnel_, no_up_, exit_;
    std::vector<char> is_pellet_, is_power_;
    std::vector<int> pellet_ids_, power_ids_, pellet_slot_, power_slot_;
    int collectibles_ = 0;
    long long sum_x_ = 0, sum_y_ = 0;
    std::vector<int> last_visit_;
    int visit_clock_ = 0;
    mutable std::vector<int> gmin_;
    mutable int path_[4][GHOST_FAR_PREDICTION_STEPS + 1] = {};
    mutable int path_length_[4] = {};
};

// The move a tree picks (train_board_graph.choose_move without random ties).
int choose_move(const Tree& tree, const Game& game, double* recently_visited_of_move) {
    int moves[4];
    double features[4][FEATURES];
    int count = game.observe(moves, features, tree.evasion);
    if (!count) return STAY;
    double scores[4];
    double best = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < count; ++i) {
        size_t at = 0;
        scores[i] = safe_number(evaluate(tree, at, features[i]));
        best = std::max(best, scores[i]);
    }
    int chosen = -1, chosen_index = -1;
    for (int i = 0; i < count; ++i) {
        if (scores[i] != best) continue;
        if (moves[i] == game.direction()) { chosen = moves[i]; chosen_index = i; break; }
        if (chosen < 0) { chosen = moves[i]; chosen_index = i; }  // lowest ACTIONS index
    }
    *recently_visited_of_move = features[chosen_index][15];
    return chosen;
}

struct Settings {
    int timestep_limit = 1000;
    double stagnation_penalty = 100, revisit_penalty = 50;
    int stagnation_grace_steps = 0, stagnation_termination_steps = 50;
    bool scaled_stagnation = false;
    double late_pellet_share = -1;  // < 0: linear
    double clear_bonus = 0.30, death_penalty = 10000, death_pellet_penalty = 0.25, size_penalty = 10;
};

double pellet_value(double ratio, double share) {
    if (share < 0) return ratio;
    if (ratio >= 1.0) return 1.0;
    const double late = 0.20, early = 1.0 - late;
    if (ratio <= early) return (1.0 - share) * ratio / early;
    return (1.0 - share) + share * ((ratio - early) / late);
}

struct Result { double fitness; double ratio; int label; int steps; long long score; };

// One training game (the per-game body of evaluate_population_lockstep).
Result play(const StaticGraph& graph, const Start& start, const Tree& tree, const Settings& s) {
    Game game(graph, start);
    int scoreless = 0, steps = 0;
    double stagnation_cost = 0;
    bool stagnated = false;
    const int total = start.pellets_eaten + start.pellets_remaining;
    for (int t = 0; t < s.timestep_limit && !game.over() && !stagnated; ++t) {
        double recently = 0;
        int move = choose_move(tree, game, &recently);
        long long before = game.score();
        game.step(move);
        ++steps;
        if (game.score() > before) {
            scoreless = 0;
        } else {
            ++scoreless;
            if (scoreless > s.stagnation_grace_steps) stagnation_cost += s.stagnation_penalty;
            if (move != STAY && recently > 0) stagnation_cost += s.revisit_penalty;
            double limit = s.scaled_stagnation
                ? static_cast<double>(s.stagnation_termination_steps) * total / std::max(1, game.pellets_remaining())
                : s.stagnation_termination_steps;
            if (scoreless >= limit) stagnated = true;
        }
    }
    int consumed = game.pellets_eaten() - start.pellets_eaten;
    double raw = static_cast<double>(consumed) / std::max(1, start.pellets_remaining);
    double ratio = pellet_value(raw, s.late_pellet_share);
    bool terminated = (game.game_over() && !game.won()) || stagnated;
    double nodes = static_cast<double>(tree.ops.size());
    double fitness = 100000.0 * ratio + 100000.0 * s.clear_bonus * game.won()
        - terminated * (s.death_penalty + s.death_pellet_penalty * 100000.0 * (1.0 - ratio))
        - stagnation_cost - s.size_penalty * nodes * nodes / 400;
    int label = game.won() ? 0 : game.game_over() ? 1 : stagnated ? 2 : 3;
    return {fitness, raw, label, steps, game.score()};
}

}  // namespace

// Python entry point: play every tree on one starting board, on several threads.
static Start make_start(
    const std::vector<int>& tiles, int width, int height,
    int player, int direction, const std::vector<int>& ghosts, const std::vector<bool>& scared,
    int powerup_turns, int pellets_eaten, int pellets_remaining, long long score,
    bool original_maze, int powerup_duration) {
    Start start;
    start.width = width; start.height = height; start.tiles = tiles;
    start.player = player; start.direction = direction;
    for (int k = 0; k < 4; ++k) { start.ghosts[k] = ghosts.at(k); start.scared[k] = scared.at(k); }
    start.powerup_turns = powerup_turns; start.pellets_eaten = pellets_eaten;
    start.pellets_remaining = pellets_remaining; start.score = score;
    start.original_maze = original_maze; start.powerup_duration = powerup_duration;
    return start;
}

// Testing aid: play one tree for up to `steps` steps and return, per step, the
// legal moves and all FEATURES features of each (FastCore/check.py --features).
py::list trace_features(
    const std::pair<std::vector<int32_t>, std::vector<double>>& tree_code,
    const std::vector<int>& tiles, int width, int height,
    int player, int direction, const std::vector<int>& ghosts, const std::vector<bool>& scared,
    int powerup_turns, int pellets_eaten, int pellets_remaining, long long score,
    bool original_maze, int powerup_duration, int steps) {
    Start start = make_start(tiles, width, height, player, direction, ghosts, scared, powerup_turns,
                             pellets_eaten, pellets_remaining, score, original_maze, powerup_duration);
    Tree tree;
    tree.ops = tree_code.first; tree.values = tree_code.second; tree.finish();
    StaticGraph graph(tiles, width, height);
    Game game(graph, start);
    py::list out;
    for (int t = 0; t < steps && !game.over(); ++t) {
        int moves[4];
        double features[4][FEATURES];
        int count = game.observe(moves, features, 1023);
        py::list step;
        for (int i = 0; i < count; ++i) {
            std::vector<double> row(features[i], features[i] + FEATURES);
            step.append(py::make_tuple(moves[i], row));
        }
        out.append(step);
        double recently = 0;
        game.step(choose_move(tree, game, &recently));
    }
    return out;
}

py::tuple evaluate_population(
    const std::vector<std::pair<std::vector<int32_t>, std::vector<double>>>& trees,
    const std::vector<int>& tiles, int width, int height,
    int player, int direction, const std::vector<int>& ghosts, const std::vector<bool>& scared,
    int powerup_turns, int pellets_eaten, int pellets_remaining, long long score,
    bool original_maze, int powerup_duration, const py::dict& settings, int threads) {
    Start start = make_start(tiles, width, height, player, direction, ghosts, scared, powerup_turns,
                             pellets_eaten, pellets_remaining, score, original_maze, powerup_duration);
    Settings s;
    s.timestep_limit = settings["timestep_limit"].cast<int>();
    s.stagnation_penalty = settings["stagnation_penalty"].cast<double>();
    s.revisit_penalty = settings["revisit_penalty"].cast<double>();
    s.stagnation_grace_steps = settings["stagnation_grace_steps"].cast<int>();
    s.stagnation_termination_steps = settings["stagnation_termination_steps"].cast<int>();
    s.scaled_stagnation = settings["scaled_stagnation"].cast<bool>();
    s.late_pellet_share = settings["late_pellet_share"].cast<double>();
    s.clear_bonus = settings["clear_bonus"].cast<double>();
    s.death_penalty = settings["death_penalty"].cast<double>();
    s.death_pellet_penalty = settings["death_pellet_penalty"].cast<double>();
    s.size_penalty = settings["size_penalty"].cast<double>();

    std::vector<Tree> compiled(trees.size());
    for (size_t i = 0; i < trees.size(); ++i) {
        compiled[i].ops = trees[i].first; compiled[i].values = trees[i].second; compiled[i].finish();
    }
    std::vector<Result> results(trees.size());
    {
        py::gil_scoped_release release;
        StaticGraph graph(tiles, width, height);
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            for (size_t i = next++; i < compiled.size(); i = next++) results[i] = play(graph, start, compiled[i], s);
        };
        int count = std::max(1, std::min<int>(threads, static_cast<int>(compiled.size())));
        std::vector<std::thread> pool;
        for (int t = 1; t < count; ++t) pool.emplace_back(worker);
        worker();
        for (auto& thread : pool) thread.join();
    }
    std::vector<double> fitness, ratio;
    std::vector<int> label, steps;
    std::vector<long long> scores;
    for (const auto& r : results) {
        fitness.push_back(r.fitness); ratio.push_back(r.ratio); label.push_back(r.label);
        steps.push_back(r.steps); scores.push_back(r.score);
    }
    return py::make_tuple(fitness, ratio, label, steps, scores);
}

PYBIND11_MODULE(_pacman_core, module) {
    module.doc() = "Fast Pac-Man training games (see FastCore/pacman_core.cpp)";
    module.def("evaluate_population", &evaluate_population,
               "Play every tree on one starting board; returns (fitness, pellet ratio, label, steps, score).");
    module.def("trace_features", &trace_features,
               "Play one tree; per step, the legal moves and their move features (for testing).");
    module.attr("FEATURES") = FEATURES;
}
