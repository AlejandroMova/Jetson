// Minimal stub — satisfies the #include in nvdsinfer_custom_impl.h.
// Our custom parser does not use any UFF functionality.
#pragma once
namespace nvuffparser {
    class IPluginFactory {};
    class IPluginFactoryExt : public IPluginFactory {};
    class IPluginFactoryV2 {};
}
