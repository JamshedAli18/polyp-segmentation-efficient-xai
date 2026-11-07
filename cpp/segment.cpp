
// segment.cpp — polyp segmentation with the ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: segment <model.onnx> <image_list.txt> <out_dir> <threads>\n";
        return 1;
    }
    const std::string model_path = argv[1], list_path = argv[2], out_dir = argv[3];
    const int threads = std::stoi(argv[4]);
    const int S = 352;
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "polyp");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path.c_str(), so);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* in_names[] = {"image"};
    const char* out_names[] = {"logits"};
    std::vector<int64_t> shape = {1, 3, S, S};
    std::vector<float> input(3 * S * S);

    std::ifstream list(list_path);
    std::string path;
    std::vector<double> times;
    while (std::getline(list, path)) {
        if (path.empty()) continue;
        cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
        const int H = bgr.rows, W = bgr.cols;

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat rgb, small;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, small, cv::Size(S, S), 0, 0, cv::INTER_LINEAR);
        for (int y = 0; y < S; ++y) {
            const cv::Vec3b* row = small.ptr<cv::Vec3b>(y);
            for (int x = 0; x < S; ++x)
                for (int c = 0; c < 3; ++c)
                    input[c * S * S + y * S + x] = (row[x][c] / 255.0f - mean[c]) / stdv[c];
        }
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                        shape.data(), shape.size());
        auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        const float* logits = out[0].GetTensorData<float>();

        cv::Mat prob(S, S, CV_32F);
        for (int k = 0; k < S * S; ++k)
            prob.at<float>(k / S, k % S) = 1.0f / (1.0f + std::exp(-logits[k]));
        cv::Mat prob_full;
        cv::resize(prob, prob_full, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        cv::Mat mask = prob_full > 0.5f;
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        std::string name = path.substr(path.find_last_of('/') + 1);
        name = name.substr(0, name.find_last_of('.')) + ".png";
        cv::imwrite(out_dir + "/" + name, mask);
    }

    std::sort(times.begin(), times.end());
    std::cout << "images " << times.size()
              << " | median " << times[times.size() / 2] << " ms"
              << " | p95 " << times[(size_t)(0.95 * (times.size() - 1))] << " ms"
              << " (end-to-end: preprocess + inference + postprocess)\n";
    return 0;
}
