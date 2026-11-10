// TapeANN v3 searcher (schema 3, written by python/build_v3.py).
//
// g++ search_v3.cpp -o search_v3 -O3 -march=native -std=c++17 -luring -lpthread
//
// Per query:
//   1. HNSW over fp16 centroids -> nearest t_max lists
//   2. keep lists with d <= (1+eps)^2 * d_nearest (at least t_min)
//   3. read their code blocks (io_uring, O_DIRECT), 4-bit fast-scan ADC
//   4. read the 4 KB vector pages of the top-R candidates, exact uint8 L2
//      over every vector on those pages, top-K

#include <immintrin.h>
#include <liburing.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

static constexpr int DIM = 128;
static constexpr int K = 10;
static constexpr int BLOCK = 32;
static constexpr size_t PAGE = 4096;
static constexpr uint32_t NO_POS = 0xFFFFFFFFu;
static constexpr uint64_t HNSW_MAGIC = 0x57534E4845504154ULL;

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// knobs
// ---------------------------------------------------------------------------
static std::string g_index_dir = ".";
static int    g_t_max = 64;
static int    g_t_min = 8;
static double g_eps = 0.3;
static int    g_rerank = 200;
static int    g_ef = 0;          // 0: max(2 * t_max, 64)
static int    g_n_queries = 0;   // 0: every query in the file
static int    g_threads = 1;
static int    g_qd = 128;
static bool   g_direct = true;
static bool   g_page_rerank = true;
static bool   g_route_check = false;
static std::string g_ids_out;
static std::string g_lat_out;

// ---------------------------------------------------------------------------
// files
// ---------------------------------------------------------------------------
static bool read_file(const std::string& path, std::vector<char>* out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "[!] cannot open " << path << "\n"; return false; }
    std::streamoff n = f.tellg();
    out->resize((size_t)n);
    f.seekg(0);
    f.read(out->data(), n);
    if (f.gcount() != n) { std::cerr << "[!] short read " << path << "\n"; return false; }
    return true;
}

template <class T>
static bool read_array(const std::string& path, std::vector<T>* out) {
    std::vector<char> raw;
    if (!read_file(path, &raw)) return false;
    if (raw.size() % sizeof(T) != 0) {
        std::cerr << "[!] " << path << " size " << raw.size() << " not a multiple of "
                  << sizeof(T) << "\n";
        return false;
    }
    out->resize(raw.size() / sizeof(T));
    std::memcpy(out->data(), raw.data(), raw.size());
    return true;
}

static long long json_int(const std::string& s, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) {
        std::cerr << "[!] v3_meta.json missing " << key << "\n";
        std::exit(1);
    }
    p = s.find(':', p) + 1;
    return std::atoll(s.c_str() + p);
}

// ---------------------------------------------------------------------------
// SIMD kernels
// ---------------------------------------------------------------------------
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline float l2_f16(const float* q, const uint16_t* c) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < DIM; i += 8) {
        __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(c + i)));
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(q + i), v);
        sum = _mm256_fmadd_ps(d, d, sum);
    }
    return hsum256(sum);
}

static inline int32_t l2_u8(const int16_t* q, const uint8_t* v) {
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < DIM; i += 16) {
        __m256i x = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(v + i)));
        __m256i d = _mm256_sub_epi16(_mm256_loadu_si256((const __m256i*)(q + i)), x);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(d, d));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// One 32-entry block: for sub-quantizer m, byte j holds entry j (low nibble)
// and entry j+16 (high nibble). lut is M x 16 uint8.
static inline void scan_block(const uint8_t* codes, const uint8_t* lut, int M,
                              __m256i* lo16, __m256i* hi16) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    for (int m = 0; m < M; ++m) {
        __m128i c = _mm_loadu_si128((const __m128i*)(codes + m * 16));
        __m128i lo = _mm_and_si128(c, mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(c, 4), mask);
        __m256i idx = _mm256_set_m128i(hi, lo);
        __m256i t = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut + m * 16)));
        __m256i r = _mm256_shuffle_epi8(t, idx);
        acc0 = _mm256_add_epi16(acc0, _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r)));
        acc1 = _mm256_add_epi16(acc1, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r, 1)));
    }
    *lo16 = acc0;
    *hi16 = acc1;
}

// ---------------------------------------------------------------------------
// index
// ---------------------------------------------------------------------------
struct ListEntry {
    uint64_t first_block;
    uint32_t n_entries;
    uint32_t n_primary;
};
static_assert(sizeof(ListEntry) == 16, "lists.bin entry is 16 bytes");

struct Hnsw {
    int max_level = 0;
    int entry = 0;
    std::vector<int32_t> cum;
    std::vector<int32_t> levels;
    std::vector<uint64_t> offsets;
    std::vector<int32_t> neighbors;

    bool load(const std::string& path) {
        std::vector<char> raw;
        if (!read_file(path, &raw)) return false;
        const char* p = raw.data();
        uint64_t magic, n;
        int32_t ml, ep, ncum, pad;
        std::memcpy(&magic, p, 8); std::memcpy(&n, p + 8, 8);
        std::memcpy(&ml, p + 16, 4); std::memcpy(&ep, p + 20, 4);
        std::memcpy(&ncum, p + 24, 4); std::memcpy(&pad, p + 28, 4);
        if (magic != HNSW_MAGIC) { std::cerr << "[!] hnsw.bin bad magic\n"; return false; }
        p += 32;
        max_level = ml;
        entry = ep;
        cum.resize(ncum); std::memcpy(cum.data(), p, ncum * 4); p += ncum * 4;
        levels.resize(n); std::memcpy(levels.data(), p, n * 4); p += n * 4;
        offsets.resize(n + 1); std::memcpy(offsets.data(), p, (n + 1) * 8); p += (n + 1) * 8;
        size_t nn = offsets[n];
        if ((size_t)(p - raw.data()) + nn * 4 != raw.size()) {
            std::cerr << "[!] hnsw.bin size mismatch\n";
            return false;
        }
        neighbors.resize(nn); std::memcpy(neighbors.data(), p, nn * 4);
        return true;
    }
    size_t bytes() const {
        return cum.size() * 4 + levels.size() * 4 + offsets.size() * 8 + neighbors.size() * 4;
    }
};

struct Index {
    int n_lists = 0;
    long long n_vectors = 0;
    int M = 0;
    int dsub = 0;
    int per_page = 30;
    size_t block_bytes = 0;
    std::vector<uint16_t> cent;   // n_lists x DIM, fp16
    std::vector<float> pq;        // M x 16 x dsub, as written
    std::vector<float> pq_t;      // M x dsub x 16
    std::vector<ListEntry> lists;
    Hnsw hnsw;
    int codes_fd = -1;
    int vec_fd = -1;
};

// ---------------------------------------------------------------------------
// per-thread state
// ---------------------------------------------------------------------------
struct Cand { float d; uint32_t pos; };

struct Stats {
    double route = 0, io = 0, scan = 0, rerank = 0;
    long long lists = 0, entries = 0, code_bytes = 0, pages = 0, cands = 0;
    double route_hits = 0;
};

struct Worker {
    const Index* ix = nullptr;
    io_uring ring{};
    std::vector<uint32_t> visit;
    uint32_t stamp = 0;
    char* buf = nullptr;
    size_t buf_cap = 0;
    std::vector<std::pair<float, int>> top;      // routed lists
    std::vector<std::pair<float, int>> cand_q;   // HNSW candidates (min-heap)
    std::vector<std::pair<float, int>> res_q;    // HNSW results (max-heap)
    std::vector<Cand> heap;                      // top-R approx, max-heap on d
    std::unordered_set<uint32_t> in_heap;
    Stats st;

    bool init(const Index* index) {
        ix = index;
        visit.assign(ix->n_lists, 0);
        if (io_uring_queue_init(std::max(g_qd, 8), &ring, 0) < 0) {
            std::cerr << "[!] io_uring init failed\n";
            return false;
        }
        return true;
    }
    void reserve(size_t bytes) {
        if (bytes <= buf_cap) return;
        if (buf) std::free(buf);
        buf_cap = std::max(bytes, buf_cap * 2);
        buf_cap = (buf_cap + PAGE - 1) / PAGE * PAGE;
        if (posix_memalign((void**)&buf, PAGE, buf_cap) != 0) {
            std::cerr << "[!] posix_memalign " << buf_cap << " failed\n";
            std::exit(1);
        }
    }

    // ---- HNSW ---------------------------------------------------------
    void route(const float* q, int ef, int k) {
        const Hnsw& h = ix->hnsw;
        auto dist = [&](int c) { return l2_f16(q, &ix->cent[(size_t)c * DIM]); };
        if (++stamp == 0) { std::fill(visit.begin(), visit.end(), 0); stamp = 1; }
        int ep = h.entry;
        float d_ep = dist(ep);
        for (int level = h.max_level; level >= 1; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                size_t b = h.offsets[ep] + h.cum[level];
                size_t e = h.offsets[ep] + h.cum[level + 1];
                for (size_t i = b; i < e; ++i) {
                    int v = h.neighbors[i];
                    if (v < 0) break;
                    float d = dist(v);
                    if (d < d_ep) { d_ep = d; ep = v; changed = true; }
                }
            }
        }
        auto cmp_min = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        };
        cand_q.clear();
        res_q.clear();
        cand_q.push_back({d_ep, ep});
        res_q.push_back({d_ep, ep});
        visit[ep] = stamp;
        while (!cand_q.empty()) {
            std::pop_heap(cand_q.begin(), cand_q.end(), cmp_min);
            auto [dc, c] = cand_q.back();
            cand_q.pop_back();
            if ((int)res_q.size() >= ef && dc > res_q.front().first) break;
            size_t b = h.offsets[c] + h.cum[0];
            size_t e = h.offsets[c] + h.cum[1];
            for (size_t i = b; i < e; ++i) {
                int v = h.neighbors[i];
                if (v < 0) break;
                if (visit[v] == stamp) continue;
                visit[v] = stamp;
                if (i + 1 < e && h.neighbors[i + 1] >= 0)
                    _mm_prefetch((const char*)&ix->cent[(size_t)h.neighbors[i + 1] * DIM],
                                 _MM_HINT_T0);
                float d = dist(v);
                if ((int)res_q.size() < ef || d < res_q.front().first) {
                    cand_q.push_back({d, v});
                    std::push_heap(cand_q.begin(), cand_q.end(), cmp_min);
                    res_q.push_back({d, v});
                    std::push_heap(res_q.begin(), res_q.end());
                    if ((int)res_q.size() > ef) {
                        std::pop_heap(res_q.begin(), res_q.end());
                        res_q.pop_back();
                    }
                }
            }
        }
        std::sort(res_q.begin(), res_q.end());
        top.assign(res_q.begin(), res_q.begin() + std::min<size_t>(k, res_q.size()));
    }

    // ---- candidates ---------------------------------------------------
    // Replicated codes return the same vector from several lists; one slot each.
    void push_cand(float d, uint32_t pos) {
        auto cmp = [](const Cand& a, const Cand& b) { return a.d < b.d; };
        const bool full = (int)heap.size() >= g_rerank;
        if (full && d >= heap.front().d) return;
        if (!in_heap.insert(pos).second) return;
        if (!full) {
            heap.push_back({d, pos});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            in_heap.erase(heap.back().pos);
            heap.back() = {d, pos};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }

    void scan_list(const float* q, int list, const char* data) {
        const int M = ix->M, dsub = ix->dsub;
        alignas(32) float r[DIM];
        const uint16_t* c = &ix->cent[(size_t)list * DIM];
        for (int i = 0; i < DIM; i += 8) {
            __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(c + i)));
            _mm256_store_ps(r + i, _mm256_sub_ps(_mm256_loadu_ps(q + i), v));
        }
        // pq_t is M x dsub x 16: table row m is two 8-wide vectors.
        alignas(32) float table[64 * 16];
        alignas(32) float mins[64];
        float bias = 0, span = 0;
        for (int m = 0; m < M; ++m) {
            const float* cb = &ix->pq_t[(size_t)m * dsub * 16];
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            for (int j = 0; j < dsub; ++j) {
                __m256 rv = _mm256_set1_ps(r[m * dsub + j]);
                __m256 d0 = _mm256_sub_ps(rv, _mm256_loadu_ps(cb + j * 16));
                __m256 d1 = _mm256_sub_ps(rv, _mm256_loadu_ps(cb + j * 16 + 8));
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
            }
            _mm256_store_ps(table + m * 16, s0);
            _mm256_store_ps(table + m * 16 + 8, s1);
            __m256 mn = _mm256_min_ps(s0, s1), mx = _mm256_max_ps(s0, s1);
            __m128 a4 = _mm_min_ps(_mm256_castps256_ps128(mn), _mm256_extractf128_ps(mn, 1));
            __m128 b4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            a4 = _mm_min_ps(a4, _mm_movehl_ps(a4, a4));
            b4 = _mm_max_ps(b4, _mm_movehl_ps(b4, b4));
            a4 = _mm_min_ss(a4, _mm_shuffle_ps(a4, a4, 1));
            b4 = _mm_max_ss(b4, _mm_shuffle_ps(b4, b4, 1));
            float lo = _mm_cvtss_f32(a4), hi = _mm_cvtss_f32(b4);
            mins[m] = lo;
            bias += lo;
            span = std::max(span, hi - lo);
        }
        if ((int)heap.size() >= g_rerank && bias >= heap.front().d) return;
        const float a = span > 0 ? 255.0f / span : 1.0f;
        alignas(32) uint8_t lut[64 * 16];
        const __m256 av = _mm256_set1_ps(a);
        for (int m = 0; m < M; ++m) {
            __m256 mv = _mm256_set1_ps(mins[m]);
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_load_ps(table + m * 16), mv), av));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_load_ps(table + m * 16 + 8), mv), av));
            __m256i w = _mm256_permute4x64_epi64(_mm256_packus_epi32(i0, i1), 0xD8);
            __m128i b = _mm_packus_epi16(_mm256_castsi256_si128(w), _mm256_extracti128_si256(w, 1));
            _mm_store_si128((__m128i*)(lut + m * 16), b);
        }

        auto threshold = [&]() -> uint16_t {
            if ((int)heap.size() < g_rerank) return 0xFFFF;
            float t = (heap.front().d - bias) * a;
            if (t < 0) return 0;
            return t >= 65535.0f ? 0xFFFF : (uint16_t)t;
        };
        const ListEntry& le = ix->lists[list];
        const int n_blocks = (int)((le.n_entries + BLOCK - 1) / BLOCK);
        const size_t code_bytes = (size_t)M * 16;
        const float inv_a = 1.0f / a;
        uint16_t thr = threshold();
        alignas(32) uint16_t acc[32];
        for (int b = 0; b < n_blocks; ++b) {
            const uint8_t* blk = (const uint8_t*)data + (size_t)b * ix->block_bytes;
            __m256i lo16, hi16;
            scan_block(blk, lut, M, &lo16, &hi16);
            __m256i tv = _mm256_set1_epi16((short)thr);
            __m256i le0 = _mm256_cmpeq_epi16(_mm256_min_epu16(lo16, tv), lo16);
            __m256i le1 = _mm256_cmpeq_epi16(_mm256_min_epu16(hi16, tv), hi16);
            uint32_t m0 = (uint32_t)_mm256_movemask_epi8(le0);
            uint32_t m1 = (uint32_t)_mm256_movemask_epi8(le1);
            if ((m0 | m1) == 0) continue;
            _mm256_store_si256((__m256i*)acc, lo16);
            _mm256_store_si256((__m256i*)(acc + 16), hi16);
            const uint32_t* pos = (const uint32_t*)(blk + code_bytes);
            uint32_t bits = _pext_u32(m0, 0x55555555u) | (_pext_u32(m1, 0x55555555u) << 16);
            while (bits) {
                int j = __builtin_ctz(bits);
                bits &= bits - 1;
                if (pos[j] == NO_POS || acc[j] > thr) continue;
                push_cand(acc[j] * inv_a + bias, pos[j]);
                thr = threshold();
            }
        }
    }

    // ---- query ----------------------------------------------------------
    struct Read { size_t off, len, buf_off; int tag; };

    // Submit reads with at most g_qd in flight; on_done(i, data) per completion.
    template <class F>
    bool run_reads(int fd, const std::vector<Read>& reads, F&& on_done, double* t_io) {
        size_t next = 0;
        int in_flight = 0;
        size_t done = 0;
        while (done < reads.size()) {
            auto t0 = Clock::now();
            int queued = 0;
            while (next < reads.size() && in_flight + queued < g_qd) {
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) break;
                const Read& r = reads[next];
                io_uring_prep_read(sqe, fd, buf + r.buf_off, (unsigned)r.len, (off_t)r.off);
                io_uring_sqe_set_data64(sqe, next);
                ++next;
                ++queued;
            }
            if (queued) {
                int n = io_uring_submit(&ring);
                if (n != queued) {
                    std::cerr << "[!] io_uring_submit " << n << " != " << queued << "\n";
                    return false;
                }
                in_flight += queued;
            }
            io_uring_cqe* cqe;
            if (io_uring_wait_cqe(&ring, &cqe) < 0) {
                std::cerr << "[!] io_uring_wait_cqe failed\n";
                return false;
            }
            std::vector<std::pair<uint64_t, int>> ready;
            unsigned head;
            io_uring_for_each_cqe(&ring, head, cqe) {
                ready.push_back({io_uring_cqe_get_data64(cqe), cqe->res});
            }
            io_uring_cq_advance(&ring, (unsigned)ready.size());
            *t_io += ms_since(t0);
            for (auto [i, res] : ready) {
                --in_flight;
                ++done;
                if (res != (int)reads[i].len) {
                    std::cerr << "[!] read " << i << " returned " << res << " of "
                              << reads[i].len << (res < 0 ? std::string(" ") + std::strerror(-res) : "")
                              << "\n";
                    return false;
                }
                on_done(i, buf + reads[i].buf_off);
            }
        }
        return true;
    }

    bool search(const float* q, uint32_t* out_ids) {
        const Index& I = *ix;
        auto t0 = Clock::now();
        const int ef = g_ef > 0 ? g_ef : std::max(2 * g_t_max, 64);
        route(q, ef, g_t_max);
        int keep = 0;
        if (!top.empty()) {
            const float lim = top[0].first * (float)((1 + g_eps) * (1 + g_eps));
            while (keep < (int)top.size() &&
                   (keep < g_t_min || top[keep].first <= lim))
                ++keep;
        }
        st.route += ms_since(t0);
        if (g_route_check) {
            std::vector<std::pair<float, int>> all(I.n_lists);
            for (int c = 0; c < I.n_lists; ++c) all[c] = {l2_f16(q, &I.cent[(size_t)c * DIM]), c};
            int T = std::min<int>(g_t_max, I.n_lists);
            std::nth_element(all.begin(), all.begin() + T, all.end());
            std::vector<char> want(I.n_lists, 0);
            for (int i = 0; i < T; ++i) want[all[i].second] = 1;
            int hit = 0;
            for (int i = 0; i < std::min<int>(T, top.size()); ++i) hit += want[top[i].second];
            st.route_hits += (double)hit / T;
        }

        std::vector<Read> reads;
        reads.reserve(keep);
        size_t cursor = 0;
        for (int i = 0; i < keep; ++i) {
            int l = top[i].second;
            const ListEntry& le = I.lists[l];
            if (le.n_entries == 0) continue;
            size_t off = le.first_block * I.block_bytes;
            size_t end = off + (size_t)((le.n_entries + BLOCK - 1) / BLOCK) * I.block_bytes;
            size_t a0 = off / PAGE * PAGE;
            size_t a1 = (end + PAGE - 1) / PAGE * PAGE;
            reads.push_back({a0, a1 - a0, cursor, l});
            cursor += a1 - a0;
            st.entries += le.n_entries;
        }
        st.lists += (long long)reads.size();
        st.code_bytes += (long long)cursor;
        reserve(std::max(cursor, (size_t)g_rerank * PAGE));

        heap.clear();
        in_heap.clear();
        double t_scan = 0;
        bool ok = run_reads(I.codes_fd, reads, [&](size_t i, const char* data) {
            auto ts = Clock::now();
            const Read& r = reads[i];
            const ListEntry& le = I.lists[r.tag];
            size_t skip = le.first_block * I.block_bytes - r.off;
            scan_list(q, r.tag, data + skip);
            t_scan += ms_since(ts);
        }, &st.io);
        if (!ok) return false;
        st.scan += t_scan;
        st.cands += (long long)heap.size();

        auto tr = Clock::now();
        std::vector<uint32_t> pages;
        pages.reserve(heap.size());
        for (const Cand& c : heap) pages.push_back(c.pos / I.per_page);
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        std::vector<Read> preads(pages.size());
        for (size_t i = 0; i < pages.size(); ++i)
            preads[i] = {(size_t)pages[i] * PAGE, PAGE, i * PAGE, (int)pages[i]};
        st.pages += (long long)pages.size();

        alignas(32) int16_t q16[DIM];
        for (int j = 0; j < DIM; ++j) q16[j] = (int16_t)std::lround(q[j]);
        std::vector<char> want;
        if (!g_page_rerank) {
            std::sort(heap.begin(), heap.end(), [](const Cand& a, const Cand& b) { return a.pos < b.pos; });
        }
        std::vector<std::pair<int32_t, uint32_t>> best;
        best.reserve(K + 1);
        auto offer = [&](int32_t d, uint32_t id) {
            if ((int)best.size() < K) {
                best.push_back({d, id});
                std::push_heap(best.begin(), best.end());
            } else if (d < best.front().first) {
                std::pop_heap(best.begin(), best.end());
                best.back() = {d, id};
                std::push_heap(best.begin(), best.end());
            }
        };
        double io_before = st.io;
        ok = run_reads(I.vec_fd, preads, [&](size_t i, const char* data) {
            const uint8_t* page = (const uint8_t*)data;
            const uint32_t* ids = (const uint32_t*)(page + (size_t)I.per_page * DIM);
            const uint32_t pg = (uint32_t)preads[i].tag;
            if (g_page_rerank) {
                for (int s = 0; s < I.per_page; ++s)
                    if (ids[s] != NO_POS) offer(l2_u8(q16, page + (size_t)s * DIM), ids[s]);
            } else {
                auto lo = std::lower_bound(heap.begin(), heap.end(), pg * (uint32_t)I.per_page,
                                           [](const Cand& c, uint32_t v) { return c.pos < v; });
                uint32_t last = NO_POS;
                for (auto it = lo; it != heap.end() && it->pos / I.per_page == pg; ++it) {
                    if (it->pos == last) continue;
                    last = it->pos;
                    int s = (int)(it->pos % I.per_page);
                    offer(l2_u8(q16, page + (size_t)s * DIM), ids[s]);
                }
            }
        }, &st.io);
        if (!ok) return false;
        st.rerank += ms_since(tr) - (st.io - io_before);

        std::sort(best.begin(), best.end());
        for (int i = 0; i < K; ++i) out_ids[i] = i < (int)best.size() ? best[i].second : NO_POS;
        return true;
    }
};

static void usage() {
    std::cerr << "usage: search_v3 --index-dir DIR [--t-max N] [--t-min N] [--eps X]\n"
              << "         [--rerank R] [--ef N] [--threads N] [--qd N] [--n-queries N]\n"
              << "         [--ids-out FILE] [--lat-out FILE] [--cache] [--no-page-rerank]\n"
              << "         [--route-check]\n";
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); std::exit(1); }
            return argv[++i];
        };
        if (a == "--index-dir") g_index_dir = need();
        else if (a == "--t-max") g_t_max = std::atoi(need());
        else if (a == "--t-min") g_t_min = std::atoi(need());
        else if (a == "--eps") g_eps = std::atof(need());
        else if (a == "--rerank") g_rerank = std::atoi(need());
        else if (a == "--ef") g_ef = std::atoi(need());
        else if (a == "--threads") g_threads = std::atoi(need());
        else if (a == "--qd") g_qd = std::atoi(need());
        else if (a == "--n-queries") g_n_queries = std::atoi(need());
        else if (a == "--ids-out") g_ids_out = need();
        else if (a == "--lat-out") g_lat_out = need();
        else if (a == "--cache") g_direct = false;
        else if (a == "--no-page-rerank") g_page_rerank = false;
        else if (a == "--route-check") g_route_check = true;
        else if (a == "--no-sqpoll") {}
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::cerr << "[!] unknown arg " << a << "\n"; usage(); return 1; }
    }
    if (g_t_min > g_t_max) g_t_min = g_t_max;
    const std::string dir = g_index_dir + "/";

    std::vector<char> meta_raw;
    if (!read_file(dir + "v3_meta.json", &meta_raw)) return 1;
    std::string meta(meta_raw.begin(), meta_raw.end());
    if (json_int(meta, "schema") != 3) { std::cerr << "[!] not a schema 3 index\n"; return 1; }

    Index ix;
    ix.n_lists = (int)json_int(meta, "n_lists");
    ix.n_vectors = json_int(meta, "n_vectors");
    ix.M = (int)json_int(meta, "pq_m");
    ix.dsub = DIM / ix.M;
    ix.per_page = (int)json_int(meta, "per_page");
    ix.block_bytes = (size_t)json_int(meta, "block_bytes");
    if (ix.M > 64 || ix.block_bytes != (size_t)ix.M * 16 + BLOCK * 4) {
        std::cerr << "[!] unsupported pq_m " << ix.M << " / block_bytes " << ix.block_bytes << "\n";
        return 1;
    }
    if (!read_array(dir + "centroids.f16", &ix.cent) ||
        !read_array(dir + "pq.f32", &ix.pq) ||
        !read_array(dir + "lists.bin", &ix.lists) ||
        !ix.hnsw.load(dir + "hnsw.bin"))
        return 1;
    if (ix.cent.size() != (size_t)ix.n_lists * DIM || ix.lists.size() != (size_t)ix.n_lists ||
        ix.pq.size() != (size_t)ix.M * 16 * ix.dsub || ix.hnsw.levels.size() != (size_t)ix.n_lists) {
        std::cerr << "[!] index files disagree with v3_meta.json\n";
        return 1;
    }
    ix.pq_t.resize(ix.pq.size());
    for (int m = 0; m < ix.M; ++m)
        for (int k = 0; k < 16; ++k)
            for (int j = 0; j < ix.dsub; ++j)
                ix.pq_t[((size_t)m * ix.dsub + j) * 16 + k] = ix.pq[((size_t)m * 16 + k) * ix.dsub + j];
    int flags = O_RDONLY | (g_direct ? O_DIRECT : 0);
    ix.codes_fd = open((dir + "codes.bin").c_str(), flags);
    ix.vec_fd = open((dir + "vectors.bin").c_str(), flags);
    if (ix.codes_fd < 0 || ix.vec_fd < 0) { perror("[!] open codes/vectors"); return 1; }

    std::vector<float> queries;
    std::vector<uint32_t> gt;
    if (!read_array(dir + "test_queries.bin", &queries) || !read_array(dir + "ground_truth.bin", &gt))
        return 1;
    const int nq_file = (int)(queries.size() / DIM);
    const int gt_k = (int)(gt.size() / nq_file);
    int nq = g_n_queries > 0 ? std::min(g_n_queries, nq_file) : nq_file;

    size_t dram = ix.cent.size() * 2 + ix.pq.size() * 4 + ix.lists.size() * sizeof(ListEntry) +
                  ix.hnsw.bytes();
    std::cout << "[*] TapeANN v3  " << ix.n_vectors << " vectors, " << ix.n_lists << " lists, PQ "
              << ix.M << "x4bit\n"
              << "[*] t_max=" << g_t_max << " t_min=" << g_t_min << " eps=" << g_eps
              << " rerank=" << g_rerank << " ef=" << (g_ef > 0 ? g_ef : std::max(2 * g_t_max, 64))
              << " threads=" << g_threads << " qd=" << g_qd
              << " page_rerank=" << g_page_rerank << " direct=" << g_direct << "\n"
              << "[*] resident index: " << dram / 1e6 << " MB\n";

    std::vector<uint32_t> ids((size_t)nq * K, NO_POS);
    std::vector<double> lat(nq, 0);
    std::vector<Worker> workers(g_threads);
    for (auto& w : workers)
        if (!w.init(&ix)) return 1;
    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    auto t_wall = Clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < g_threads; ++t) {
        pool.emplace_back([&, t]() {
            Worker& w = workers[t];
            for (int qi = next++; qi < nq && !failed; qi = next++) {
                auto t0 = Clock::now();
                if (!w.search(&queries[(size_t)qi * DIM], &ids[(size_t)qi * K])) failed = true;
                lat[qi] = ms_since(t0);
            }
        });
    }
    for (auto& th : pool) th.join();
    const double wall_s = ms_since(t_wall) / 1000.0;
    if (failed) { std::cerr << "[!] search failed\n"; return 1; }

    Stats s;
    for (auto& w : workers) {
        s.route += w.st.route; s.io += w.st.io; s.scan += w.st.scan; s.rerank += w.st.rerank;
        s.lists += w.st.lists; s.entries += w.st.entries; s.code_bytes += w.st.code_bytes;
        s.pages += w.st.pages; s.cands += w.st.cands; s.route_hits += w.st.route_hits;
    }
    double hits = 0;
    for (int qi = 0; qi < nq; ++qi) {
        const uint32_t* g = &gt[(size_t)qi * gt_k];
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                if (ids[(size_t)qi * K + i] == g[j]) { hits += 1; break; }
    }
    std::vector<double> sorted = lat;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted[std::min<size_t>(sorted.size() - 1, (size_t)(p * sorted.size()))]; };
    double mean = 0;
    for (double v : lat) mean += v;
    mean /= nq;

    std::cout << "Queries          : " << nq << "\n"
              << "Avg lists        : " << (double)s.lists / nq << "\n"
              << "Avg entries      : " << (double)s.entries / nq << "\n"
              << "Avg candidates   : " << (double)s.cands / nq << "\n"
              << "Avg pages        : " << (double)s.pages / nq << "\n"
              << "Avg bytes read   : " << (double)(s.code_bytes + s.pages * PAGE) / nq << "\n"
              << "Avg latency      : " << mean << " ms\n"
              << "  p50 / p95 / p99: " << pct(0.5) << " / " << pct(0.95) << " / " << pct(0.99) << " ms\n"
              << "  E1 t_route     : " << s.route / nq << " ms\n"
              << "  E1 t_io        : " << s.io / nq << " ms\n"
              << "  E1 t_scan      : " << s.scan / nq << " ms\n"
              << "  E1 t_rerank    : " << s.rerank / nq << " ms\n";
    if (g_route_check)
        std::cout << "Route recall@T   : " << 100.0 * s.route_hits / nq << " %\n";
    std::cout << "QPS              : " << nq / wall_s << "\n"
              << "Recall@10        : " << 100.0 * hits / (nq * K) << " %\n";

    if (!g_ids_out.empty()) {
        FILE* f = std::fopen(g_ids_out.c_str(), "wb");
        if (!f || std::fwrite(ids.data(), 4, ids.size(), f) != ids.size()) { perror("[!] ids-out"); return 1; }
        std::fclose(f);
    }
    if (!g_lat_out.empty()) {
        FILE* f = std::fopen(g_lat_out.c_str(), "wb");
        if (!f || std::fwrite(lat.data(), 8, lat.size(), f) != lat.size()) { perror("[!] lat-out"); return 1; }
        std::fclose(f);
    }
    return 0;
}
