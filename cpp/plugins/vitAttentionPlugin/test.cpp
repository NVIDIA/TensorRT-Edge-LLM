#include <cstdint>

using namespace nvinfer1;

namespace trt_edgellm
{  
namespace plugins
{
    namespace {
        constexpr char const* kVIT_ATTENTION_PLUGIN_VERSION{"1"};
        constexpr char const* kVIT_ATTENTION_PLUGIN_NAME{"ViTAttentionPlugin"};

        void applyThorSMRenumberWAR(int32_t& smVersion)
        {
            if (smVersion == 110)
            {
                smVersion = 101;
            }
        }
    }
}