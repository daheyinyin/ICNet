

#include "ICNet.h"
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NUM = 19;
    const uint32_t MODEL_TYPE = 1;
    const uint32_t FRAMEWORK_TYPE = 2;
}

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './icnet test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.modelType = MODEL_TYPE;
    initParam.labelPath = "../data/models/icnet.names";
    initParam.modelPath = "../data/models/icnet_tran.om";
    initParam.checkModel = true;
    initParam.frameworkType = FRAMEWORK_TYPE;

    ICNet model;
    APP_ERROR ret = model.Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "ICNet init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    ret = model.Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "ICNet process failed, ret=" << ret << ".";
        model.DeInit();
        return ret;
    }
    model.DeInit();
    return APP_ERR_OK;
}
