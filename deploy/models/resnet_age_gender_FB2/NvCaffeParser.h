// Minimal stub — satisfies the #include in nvdsinfer_custom_impl.h.
// Our custom parser does not use any Caffe functionality.
#pragma once
namespace nvcaffeparser1 {
    class IPluginFactory {};
    class IPluginFactoryExt : public IPluginFactory {};
    class IPluginFactoryV2 {};
}
