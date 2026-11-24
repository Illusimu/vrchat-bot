#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <openvr.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <io.h>
#include <vector>

using Microsoft::WRL::ComPtr;

struct EyeTexture {
    vr::IVRCompositor* compositor = nullptr;
    void* mirror = nullptr;
    ComPtr<ID3D11Texture2D> texture, staging;
    D3D11_TEXTURE2D_DESC desc{};
    bool bgra = false;

    ~EyeTexture() {
        if (mirror) compositor->ReleaseMirrorTextureD3D11(mirror);
    }

    int open(vr::IVRCompositor* source, ID3D11Device* device, vr::EVREye eye) {
        compositor = source;
        if (source->GetMirrorTextureD3D11(eye, device, &mirror)
            != vr::VRCompositorError_None || !mirror) return 5;
        ComPtr<ID3D11Resource> resource;
        static_cast<ID3D11ShaderResourceView*>(mirror)->GetResource(&resource);
        if (FAILED(resource.As(&texture))) return 6;
        texture->GetDesc(&desc);
        bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
            || desc.Format == DXGI_FORMAT_B8G8R8A8_TYPELESS;
        if (!bgra && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
            && desc.Format != DXGI_FORMAT_R8G8B8A8_TYPELESS) return 7;
        auto readback = desc;
        readback.BindFlags = readback.MiscFlags = 0;
        readback.Usage = D3D11_USAGE_STAGING;
        readback.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        return FAILED(device->CreateTexture2D(&readback, nullptr, &staging)) ? 8 : 0;
    }
};

bool hasPicture(const D3D11_MAPPED_SUBRESOURCE& pixels, UINT width, UINT height) {
    auto* data = static_cast<const unsigned char*>(pixels.pData);
    for (UINT y = 0; y < height; y += 16) {
        auto* row = data + y * pixels.RowPitch;
        for (UINT x = 0; x < width; x += 16) {
            auto* rgb = row + x * 4;
            if (rgb[0] > 8 || rgb[1] > 8 || rgb[2] > 8) return true;
        }
    }
    return false;
}

int capture(vr::IVRSystem* system, bool stereo) {
    int32_t index = -1;
    system->GetDXGIOutputInfo(&index);
    ComPtr<IDXGIFactory1> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapters(index, &adapter))) return 2;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    if (FAILED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &context))) return 3;
    auto compositor = vr::VRCompositor();
    if (!compositor) return 4;
    EyeTexture eyes[2];
    const unsigned eyeCount = stereo ? 2 : 1;
    for (unsigned i = 0; i < eyeCount; ++i) {
        if (auto result = eyes[i].open(compositor, device.Get(), i ? vr::Eye_Right : vr::Eye_Left))
            return result;
    }
    const auto& left = eyes[0];
    const UINT width = left.desc.Width, height = left.desc.Height;
    if (stereo && (eyes[1].desc.Width != width || eyes[1].desc.Height != height
        || eyes[1].bgra != left.bgra)) return 7;

    // SteamVR can return a blank shared texture before it draws the first mirror frame.
    // Keep both subscriptions alive and queue their copies together before either readback.
    const auto deadline = GetTickCount64() + 900;
    do {
        Sleep(50);
        for (unsigned i = 0; i < eyeCount; ++i)
            context->CopyResource(eyes[i].staging.Get(), eyes[i].texture.Get());
        D3D11_MAPPED_SUBRESOURCE pixels[2]{};
        unsigned mapped = 0;
        while (mapped < eyeCount && SUCCEEDED(context->Map(
            eyes[mapped].staging.Get(), 0, D3D11_MAP_READ, 0, &pixels[mapped]))) ++mapped;
        bool ready = mapped == eyeCount;
        for (unsigned i = 0; ready && i < eyeCount; ++i)
            ready = hasPicture(pixels[i], width, height);

        std::vector<unsigned char> frame;
        if (ready) {
            const size_t rowBytes = size_t(width) * 4;
            frame.resize(rowBytes * height * eyeCount);
            for (UINT y = 0; y < height; ++y)
                for (unsigned i = 0; i < eyeCount; ++i)
                    std::memcpy(frame.data() + (y * eyeCount + i) * rowBytes,
                        static_cast<const unsigned char*>(pixels[i].pData) + y * pixels[i].RowPitch,
                        rowBytes);
        }
        for (unsigned i = 0; i < mapped; ++i)
            context->Unmap(eyes[i].staging.Get(), 0);
        if (mapped != eyeCount) return 9;
        if (ready) {
            _setmode(_fileno(stdout), _O_BINARY);
            printf("%u %u %s\n", width * eyeCount, height, left.bgra ? "BGRA" : "RGBA");
            return fwrite(frame.data(), 1, frame.size(), stdout) == frame.size() ? 0 : 11;
        }
    } while (GetTickCount64() < deadline);
    fputs("SteamVR mirror textures did not produce a visible frame\n", stderr);
    return 10;
}

int main(int argc, char** argv) {
    const bool stereo = argc == 2 && !std::strcmp(argv[1], "--stereo");
    if (argc != 1 && !stereo) {
        fputs("Usage: vr_eye [--stereo]\n", stderr);
        return 1;
    }
    vr::EVRInitError error;
    auto system = vr::VR_Init(&error, vr::VRApplication_Background);
    if (error) { fprintf(stderr, "%s", vr::VR_GetVRInitErrorAsEnglishDescription(error)); return 1; }
    auto result = capture(system, stereo);
    vr::VR_Shutdown();
    return result;
}
