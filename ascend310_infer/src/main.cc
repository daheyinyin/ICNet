

#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "../inc/utils.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/include/vision_ascend.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using mindspore::Serialization;
using mindspore::Model;
using mindspore::Context;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::Graph;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Rescale;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::HWC2CHW;

using mindspore::dataset::transforms::TypeCast;

DEFINE_string(model_path, "/root/ICNet.mindir", "model path");
DEFINE_string(dataset_path, "/data/cityscapes/leftImg8bit/val", "dataset path");
DEFINE_int32(input_width, 2048, "input width");
DEFINE_int32(input_height, 1024, "inputheight");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision_mode, "allow_fp32_to_fp16", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(device_target, "Ascend310", "device target");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_model_path).empty()) {
        std::cout << "Invalid model" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    Graph graph;
    Status ret = Serialization::Load(FLAGS_model_path, ModelType::kMindIR, &graph);
    if (ret != kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }

    Model model;
    ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> modelInputs = model.GetInputs();

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    auto decode = Decode();
    auto normalize = Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    auto hwc2chw = HWC2CHW();
    auto rescale = Rescale(1.0 / 255.0, 0);
    auto typeCast = TypeCast("float32");

    mindspore::dataset::Execute transformDecode(decode);
    mindspore::dataset::Execute transform({rescale, normalize, hwc2chw});
    mindspore::dataset::Execute transformCast(typeCast);

    std::map<double, double> costTime_map;

    size_t size = all_files.size();
    for (size_t i = 0; i < size; ++i) {
        struct timeval start;
        struct timeval end;
        double startTime_ms;
        double endTime_ms;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;

        std::cout << "Start predict input files:" << all_files[i] << std::endl;
        mindspore::MSTensor image =  ReadFileToTensor(all_files[i]);

        transformDecode(image, &image);
        std::vector<int64_t> shape = image.Shape();
        transform(image, &image);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            image.Data().get(), image.DataSize());

        gettimeofday(&start, NULL);
        model.Predict(inputs, &outputs);
        gettimeofday(&end, NULL);

        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        WriteResult(all_files[i], outputs);
    }
    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }

    average = average / infer_cnt;

    std::stringstream timeCost;
    timeCost << "NN inference cost average time: " << average << " ms of infer_count " << infer_cnt << std::endl;
    std::cout << "NN inference cost average time: " << average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << timeCost.str();
    file_stream.close();
    costTime_map.clear();
    return 0;
}
