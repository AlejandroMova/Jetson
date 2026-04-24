/*
 * Custom classifier parser for ResNet-18 Age/Gender (TAO classification_pyt).
 * TAO exports the model WITHOUT a final Softmax layer — the output tensor
 * contains raw logits. This parser applies Softmax before writing to
 * NvDsClassifierMeta so that result_prob is always in [0.0, 1.0].
 *
 * Compile (inside the DeepStream container):
 *   g++ -shared -fPIC -o libcustom_softmax_parser.so custom_softmax_parser.cpp \
 *       -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *       -std=c++14 -O2
 */

#include "nvdsinfer.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>

static const std::vector<std::string> kLabels = {
    "female_adult",
    "female_senior",
    "female_young",
    "male_adult",
    "male_senior",
    "male_young"
};

extern "C"
bool CustomClassifierParseFunction(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo  const& networkInfo,
    float                        classifierThreshold,
    std::vector<NvDsInferAttribute>& attrList,
    std::string&                 descString)
{
    if (outputLayersInfo.empty())
        return false;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const unsigned int numClasses   = static_cast<unsigned int>(kLabels.size());

    if (layer.inferDims.numElements != numClasses)
        return false;

    const float* logits = static_cast<const float*>(layer.buffer);

    /* --- Numerically stable Softmax --- */
    float maxLogit = *std::max_element(logits, logits + numClasses);

    float sumExp = 0.0f;
    std::vector<float> expVals(numClasses);
    for (unsigned int i = 0; i < numClasses; ++i) {
        expVals[i] = std::exp(logits[i] - maxLogit);
        sumExp    += expVals[i];
    }

    /* --- Find argmax --- */
    unsigned int bestIdx  = 0;
    float        bestProb = 0.0f;
    for (unsigned int i = 0; i < numClasses; ++i) {
        float prob = expVals[i] / sumExp;
        if (prob > bestProb) {
            bestProb = prob;
            bestIdx  = i;
        }
    }

    /* --- Apply classifier threshold --- */
    if (bestProb < classifierThreshold)
        return true;   /* no output — below threshold */

    NvDsInferAttribute attr;
    attr.attributeIndex      = 0;
    attr.attributeValue      = bestIdx;
    attr.attributeConfidence = bestProb;                     /* guaranteed 0-1 */
    attr.attributeLabel      = strdup(kLabels[bestIdx].c_str()); /* freed by DS */

    attrList.push_back(attr);
    descString = kLabels[bestIdx];

    return true;
}
