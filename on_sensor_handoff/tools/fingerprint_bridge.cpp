#include "camera/lightweight_fingerprint.hpp"

#include <cstdint>

// Build this wrapper with Pxl-loc-tak's original implementation. Keeping the
// implementation external makes this baseline match the existing Pi code.
extern "C" int fingerprint_yuv420(
    const std::uint8_t* frame, int width, int height, int stride,
    int x, int y, int box_width, int box_height, float* output,
    int output_capacity) {
    if (!output || output_capacity < 140) return -1;
    const auto descriptor = imx500_bridge::yuv420AppearanceDescriptor(
        frame, width, height, stride, x, y, box_width, box_height);
    if (descriptor.size() != 140) return -2;
    for (int index = 0; index < 140; ++index) output[index] = descriptor[index];
    return 140;
}
