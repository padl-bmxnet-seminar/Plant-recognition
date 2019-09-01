/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 *
 * This is a simple predictor which shows how to use c api for image classification. It uses
 * opencv for image reading.
 *
 * Created by liuxiao on 12/9/15.
 * Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
 * Home Page: www.liuxiao.org
 * E-mail: liuxiao@foxmail.com
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
#include <future>
#include <opencv2/opencv.hpp>
// Path for c_predict_api
#include <mxnet/c_predict_api.h>


const mx_float DEFAULT_MEAN = 117.0;

static std::string trim(const std::string &input) {
    auto not_space = [](int ch) {
        return !std::isspace(ch);
    };
    auto output = input;
    output.erase(output.begin(), std::find_if(output.begin(), output.end(), not_space));
    output.erase(std::find_if(output.rbegin(), output.rend(), not_space).base(), output.end());
    return output;
}

// Read file to buffer
class BufferFile {
public :
    std::string file_path_;
    std::size_t length_ = 0;
    std::unique_ptr<char[]> buffer_;

    explicit BufferFile(const std::string &file_path)
            : file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = static_cast<std::size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

        buffer_.reset(new char[length_]);
        ifs.read(buffer_.get(), length_);
        ifs.close();
    }

    std::size_t GetLength() {
        return length_;
    }

    char *GetBuffer() {
        return buffer_.get();
    }
};

void GetImageFile(const std::string &image_file,
                  mx_float *image_data, int channels,
                  cv::Size resize_size, const mx_float *mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float *ptr_image_r = image_data;
    mx_float *ptr_image_g = image_data + size / 3;
    mx_float *ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++) {
        auto data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
                mean_data++;
            }
            if (channels > 1) {
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector <std::string> LoadSynset(const std::string &synset_file) {
    std::ifstream fi(synset_file.c_str());

    if (!fi.is_open()) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector <std::string> output;

    std::string synset, lemma;
    while (fi >> synset) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void split(const std::string &s, char delim, std::vector <std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

void splitValFile(std::vector<float> &labels, std::vector <std::string> &paths) {
    std::ifstream infile("../../../../binary_plant_recognition/Data/mushrooms_9class/data_list_val_copy.lst");

    std::string line;

    while (std::getline(infile, line)) {
        std::vector <std::string> row_values;

        split(line, '\t', row_values);
        labels.push_back(std::atof(row_values[1].c_str()));
        paths.push_back(row_values[2]);

        std::cout << std::endl;
    }
}

void PrintOutputResult(const std::vector<float> &data, const std::vector <std::string> &synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size do not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    std::size_t best_idx = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        std::cout << "Accuracy[" << i << "] = " << std::setprecision(8) << data[i] << std::endl;

        if (data[i] > best_accuracy) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

    std::cout << "Best Result: " << trim(synset[best_idx]) << " (id=" << best_idx << ", " <<
              "accuracy=" << std::setprecision(8) << best_accuracy << ")" << std::endl;
}

std::vector<float> predict(PredictorHandle pred_hnd, const std::vector <mx_float> &image_data,
                           NDListHandle nd_hnd, const std::string &synset_file, int i) {
    auto image_size = image_data.size();
    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));

    // Do Predict Forward
    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint *shape = nullptr;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    std::size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

    std::vector<float> data(size);

    MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));

    // Release NDList
    if (nd_hnd) {
        MXNDListFree(nd_hnd);
    }

    return data;
}

void calcAcc(std::vector<float> preds, std::vector<float> labels) {
    if (preds.size() != labels.size()) {
        std::cout << "Number of preds differ from number of labels" << std::endl;
        return;
    }
    float sum = 0;
    for (int i = 0; i < preds.size(); i++) {
        if (preds[i] == labels[i]) {
            sum++;
        }
    }
    float result = (float) (sum / preds.size());
    std::cout << "Acc: " << result << std::endl;
}

std::vector<float> getBest(std::vector <std::vector<float>> &vec) {
    std::vector<float> result;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        float best_accuracy = 0;
        float best_idx = 0;
        for (int j = 0; j < vec[i].size(); ++j) {
            if (vec[i][j] > best_accuracy) {
                best_accuracy = vec[i][j];
                best_idx = (float) j;
            }
        }
        result.push_back(best_idx);
    }
    return result;
}

void validateModel() {
    std::string json_file = "/home/padl19t4/binary_plant_recognition/image-classifier-1bit-symbol.json";
    std::string param_file = "../../../../binary_plant_recognition/image-classifier-1bit-0000.params";
    std::string synset_file = "../../../../binary_plant_recognition/mushrooms9_synset.txt";
    std::string nd_file = "model/Inception/mean_224.nd";
    std::vector<float> labels;
    std::vector <std::string> paths;
    splitValFile(labels, paths);
    BufferFile json_data(json_file);
    BufferFile param_data(param_file);
    std::string prefix = "/home/padl19t4/binary_plant_recognition/Data/mushrooms_9class/";
    int num_threads = 5;

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char *input_key[1] = {"data"};
    const char **input_keys = input_key;

    // Image size and channels
    int width = 336;
    int height = 336;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = {0, 4};
    const mx_uint input_shape_data[4] = {1,
                                         static_cast<mx_uint>(channels),
                                         static_cast<mx_uint>(height),
                                         static_cast<mx_uint>(width)};


    auto image_size = static_cast<std::size_t>(width * height * channels);

    // Read Mean Data
    const mx_float *nd_data = nullptr;
    NDListHandle nd_hnd = nullptr;
    BufferFile nd_buf(nd_file);

    // Read Image Data
    //std::vector <mx_float> image_data(image_size);
    std::vector <std::vector<mx_float>> images;
    for (int i = 0; i < paths.size(); i++) {
        std::vector <mx_float> image_data(image_size);
        images.push_back(image_data);
        std::string path = prefix + paths[i];
        std::cout << path << std::endl;
        GetImageFile(path, images[i].data(), channels, cv::Size(width, height), nd_data);
    }
    std::vector <std::vector<float>> results;
    using namespace std;
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < paths.size(); i++) {
            cout<<i<<endl;
            if (num_threads == 1) {
                // Create Predictor
                PredictorHandle pred_hnd;
                MXPredCreate(static_cast<const char *>(json_data.GetBuffer()),
                             static_cast<const char *>(param_data.GetBuffer()),
                             static_cast<int>(param_data.GetLength()),
                             dev_type,
                             dev_id,
                             num_input_nodes,
                             input_keys,
                             input_shape_indptr,
                             input_shape_data,
                             &pred_hnd);
                assert(pred_hnd);

                results.push_back(predict(pred_hnd, images[i], nd_hnd, synset_file, 0));
            } else {
                // Create Predictor
                std::cout << i << std::endl;
                std::vector <PredictorHandle> pred_hnds(num_threads, nullptr);
                MXPredCreateMultiThread(static_cast<const char *>(json_data.GetBuffer()),
                                        static_cast<const char *>(param_data.GetBuffer()),
                                        static_cast<int>(param_data.GetLength()),
                                        dev_type,
                                        dev_id,
                                        num_input_nodes,
                                        input_keys,
                                        input_shape_indptr,
                                        input_shape_data,
                                        pred_hnds.size(),
                                        pred_hnds.data());
                for (auto hnd : pred_hnds)
                    assert(hnd);
                std::vector < std::future < std::vector < float >> > futures;
                std::vector <std::thread> threads;
                std::vector<float> tmp_result;
                for (int j = 0; j < num_threads; j++)
                    futures.push_back(std::async(predict, pred_hnds[j], images[i], nd_hnd, synset_file, j));
                //threads.emplace_back(predict, pred_hnds[j], images[i], nd_hnd, synset_file, j);
                for (int j = 0; j < num_threads; j++) {
                    tmp_result = futures[j].get();
                    MXPredFree(pred_hnds[j]);

                }
                results.push_back(tmp_result);
                //threads[j].join();
            }
        }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "elapsed time: " << elapsed.count() << std::endl;
    std::vector<float> best = getBest(results);
    calcAcc(best, labels);
    printf("run successfully\n");
}

int main(int argc, char *argv[]) {
    validateModel();
    return EXIT_SUCCESS;
}
