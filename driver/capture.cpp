#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <openvr.h>
#include <cstdio>
#include <fcntl.h>
#include <io.h>

using Microsoft::WRL::ComPtr;

int capture(vr::IVRSystem* system) {
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
    void* mirror = nullptr;
    if (compositor->GetMirrorTextureD3D11(vr::Eye_Left, device.Get(), &mirror)
        != vr::VRCompositorError_None || !mirror) return 5;
    ComPtr<ID3D11Resource> resource;
    static_cast<ID3D11ShaderResourceView*>(mirror)->GetResource(&resource);
    ComPtr<ID3D11Texture2D> texture;
    auto result = resource.As(&texture);
    compositor->ReleaseMirrorTextureD3D11(mirror);
    if (FAILED(result)) return 6;
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);
    bool bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        || desc.Format == DXGI_FORMAT_B8G8R8A8_TYPELESS;
    if (!bgra && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        && desc.Format != DXGI_FORMAT_R8G8B8A8_TYPELESS) return 7;
    desc.BindFlags = desc.MiscFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, &staging))) return 8;
    context->CopyResource(staging.Get(), texture.Get());
    D3D11_MAPPED_SUBRESOURCE pixels{};
    if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &pixels))) return 9;
    _setmode(_fileno(stdout), _O_BINARY);
    printf("%u %u %s\n", desc.Width, desc.Height, bgra ? "BGRA" : "RGBA");
    for (UINT y = 0; y < desc.Height; ++y)
        fwrite(static_cast<char*>(pixels.pData) + y * pixels.RowPitch, 4, desc.Width, stdout);
    context->Unmap(staging.Get(), 0);
    return 0;
}

int main() {
    vr::EVRInitError error;
    auto system = vr::VR_Init(&error, vr::VRApplication_Background);
    if (error) { fprintf(stderr, "%s", vr::VR_GetVRInitErrorAsEnglishDescription(error)); return 1; }
    auto result = capture(system);
    vr::VR_Shutdown();
    return result;
}
