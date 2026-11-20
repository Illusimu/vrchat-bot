#include <winsock2.h>
#include <openvr_driver.h>
#include "hand_simulation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
using Clock = std::chrono::steady_clock;
constexpr unsigned short port = 29745;
constexpr auto inputTimeout = std::chrono::milliseconds(250);
constexpr size_t poseSize = 7;
constexpr size_t poseCount = 11;
constexpr size_t handSize = 11;
constexpr size_t payloadSize = poseSize * poseCount + handSize * 2;
constexpr size_t packetSize = 4 + payloadSize * sizeof(float);
constexpr uint64_t universe = 41861104;

constexpr const char* chaperone = R"({
  "json_id":"chaperone_info","version":5,"time":"2026-11-06T00:00:00Z",
  "universes":[{"universeID":"41861104","play_area":[2,2],"collision_bounds":[],
    "standing":{"translation":[0,0,0],"yaw":0},
    "seated":{"translation":[0,0,0],"yaw":0}}]
})";

class Display final : public vr::IVRDisplayComponent {
    void GetWindowBounds(int32_t* x, int32_t* y, uint32_t* w, uint32_t* h) override {
        *x = 0; *y = 0; *w = 1280; *h = 720;
    }
    bool IsDisplayOnDesktop() override { return true; }
    bool IsDisplayRealDisplay() override { return false; }
    void GetRecommendedRenderTargetSize(uint32_t* w, uint32_t* h) override {
        *w = 960; *h = 960;
    }
    void GetEyeOutputViewport(vr::EVREye eye, uint32_t* x, uint32_t* y,
                              uint32_t* w, uint32_t* h) override {
        *x = eye == vr::Eye_Left ? 0 : 640; *y = 0; *w = 640; *h = 720;
    }
    void GetProjectionRaw(vr::EVREye, float* left, float* right,
                          float* top, float* bottom) override {
        *left = -1; *right = 1; *top = -1; *bottom = 1;
    }
    vr::DistortionCoordinates_t ComputeDistortion(vr::EVREye, float u, float v) override {
        return {{u, v}, {u, v}, {u, v}};
    }
    bool ComputeInverseDistortion(vr::HmdVector2_t* result, vr::EVREye, uint32_t,
                                  float u, float v) override {
        *result = {{u, v}};
        return true;
    }
};

struct Component {
    const char* path;
    bool button = false;
    bool signedAxis = false;
    vr::VRInputComponentHandle_t handle = vr::k_ulInvalidInputComponentHandle;
};

class Device final : public vr::ITrackedDeviceServerDriver {
public:
    Device(const char* serial, vr::ETrackedDeviceClass kind,
           vr::ETrackedControllerRole role = vr::TrackedControllerRole_Invalid)
        : serial(serial), kind(kind), role(role) {}

    vr::EVRInitError Activate(uint32_t index) override {
        this->index = index;
        auto* props = vr::VRProperties();
        const auto container = props->TrackedDeviceToPropertyContainer(index);
        char registeredName[96];
        snprintf(registeredName, sizeof(registeredName), "desktopbody/%s", serial);
        props->SetStringProperty(container, vr::Prop_RegisteredDeviceType_String, registeredName);
        props->SetStringProperty(container, vr::Prop_TrackingSystemName_String, "desktopbody");
        props->SetStringProperty(container, vr::Prop_SerialNumber_String, serial);
        props->SetStringProperty(container, vr::Prop_ManufacturerName_String, "Desktop Body");
        props->SetUint64Property(container, vr::Prop_CurrentUniverseId_Uint64, universe);

        switch (kind) {
        case vr::TrackedDeviceClass_HMD:
            props->SetStringProperty(container, vr::Prop_ModelNumber_String, "Desktop Body HMD");
            props->SetStringProperty(container, vr::Prop_RenderModelName_String, "generic_hmd");
            props->SetFloatProperty(container, vr::Prop_UserIpdMeters_Float, 0.064f);
            props->SetFloatProperty(container, vr::Prop_DisplayFrequency_Float, 90.f);
            props->SetFloatProperty(container, vr::Prop_SecondsFromVsyncToPhotons_Float, 1.f / 90.f);
            props->SetBoolProperty(container, vr::Prop_DisplayDebugMode_Bool, true);
            props->SetBoolProperty(container, vr::Prop_IsOnDesktop_Bool, true);
            props->SetStringProperty(container, vr::Prop_DriverProvidedChaperoneJson_String, chaperone);
            break;
        case vr::TrackedDeviceClass_Controller:
            if (!ActivateHand(container)) {
                vr::VRDriverLog()->Log("Desktop Body: failed to create Index input components");
                Deactivate();
                return vr::VRInitError_Driver_Failed;
            }
            break;
        default:
            props->SetStringProperty(container, vr::Prop_ModelNumber_String, serial);
            props->SetStringProperty(container, vr::Prop_ControllerType_String, "vive_tracker");
            props->SetStringProperty(container, vr::Prop_RenderModelName_String, "{htc}vr_tracker_vive_1_0");
            props->SetStringProperty(container, vr::Prop_InputProfilePath_String,
                                     "{htc}/input/vive_tracker_profile.json");
            break;
        }
        return vr::VRInitError_None;
    }

    void Deactivate() override { index = vr::k_unTrackedDeviceIndexInvalid; }
    void EnterStandby() override {}
    void DebugRequest(const char*, char* response, uint32_t length) override {
        if (length) response[0] = '\0';
    }
    void* GetComponent(const char* name) override {
        if (kind == vr::TrackedDeviceClass_HMD && !strcmp(name, vr::IVRDisplayComponent_Version)) {
            return static_cast<vr::IVRDisplayComponent*>(&display);
        }
        return nullptr;
    }
    vr::DriverPose_t GetPose() override { return {}; }

    void PublishPose(const float* values, Clock::time_point stamp, Clock::time_point now) {
        if (index == vr::k_unTrackedDeviceIndexInvalid) return;
        vr::DriverPose_t pose{};
        std::copy_n(values, 3, pose.vecPosition);
        const double norm = std::sqrt(values[3] * values[3] + values[4] * values[4] +
                                      values[5] * values[5] + values[6] * values[6]);
        pose.qRotation = {values[3] / norm, values[4] / norm, values[5] / norm, values[6] / norm};
        if (stamp != previousStamp) {
            const double dt = std::chrono::duration<double>(stamp - previousStamp).count();
            for (size_t i = 0; i < 3; ++i) {
                linearVelocity[i] = previousStamp != Clock::time_point{} && dt > .001 && dt < .15
                    ? (pose.vecPosition[i] - previousPosition[i]) / dt : 0.;
                previousPosition[i] = pose.vecPosition[i];
            }
            previousStamp = stamp;
        }
        if (now - stamp < std::chrono::milliseconds(70))
            std::copy(linearVelocity.begin(), linearVelocity.end(), pose.vecVelocity);
        pose.qWorldFromDriverRotation.w = pose.qDriverFromHeadRotation.w = 1;
        pose.poseIsValid = pose.deviceIsConnected = true;
        pose.result = vr::TrackingResult_Running_OK;
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index, pose, sizeof(pose));
    }

    void PublishHand(const float* values) {
        if (index == vr::k_unTrackedDeviceIndexInvalid) return;
        const float x = values[0], y = values[1], trigger = values[2], grip = values[3];
        const float a = values[4], b = values[5];
        const bool stickActive = x != 0 || y != 0;
        const float inputs[] = {x, y, stickActive ? 1.f : 0.f, trigger,
            trigger > 0.75f ? 1.f : 0.f, trigger, grip, grip, grip, a, a, b, b};
        static_assert(std::size(inputs) == 13);
        for (size_t i = 0; i < components.size(); ++i) {
            const auto& component = components[i];
            if (component.button) {
                vr::VRDriverInput()->UpdateBooleanComponent(component.handle, inputs[i] > 0, 0);
            } else {
                vr::VRDriverInput()->UpdateScalarComponent(component.handle, inputs[i], 0);
            }
        }
        const MyFingerCurls curls{values[6], values[7], values[8], values[9], values[10]};
        vr::VRBoneTransform_t bones[eBone_Count]{};
        hand.ComputeSkeletonTransforms(role, curls, {}, bones);
        vr::VRDriverInput()->UpdateSkeletonComponent(skeleton, vr::VRSkeletalMotionRange_WithController,
                                                     bones, eBone_Count);
        vr::VRDriverInput()->UpdateSkeletonComponent(skeleton, vr::VRSkeletalMotionRange_WithoutController,
                                                     bones, eBone_Count);
    }

    const char* const serial;
    const vr::ETrackedDeviceClass kind;

private:
    bool ActivateHand(vr::PropertyContainerHandle_t container) {
        const bool left = role == vr::TrackedControllerRole_LeftHand;
        auto* props = vr::VRProperties();
        props->SetStringProperty(container, vr::Prop_ModelNumber_String,
                                 left ? "Desktop Body Left Hand" : "Desktop Body Right Hand");
        props->SetStringProperty(container, vr::Prop_ControllerType_String, "knuckles");
        props->SetStringProperty(container, vr::Prop_InputProfilePath_String,
                                 "{indexcontroller}/input/index_controller_profile.json");
        props->SetStringProperty(container, vr::Prop_RenderModelName_String,
                                 left ? "{indexcontroller}valve_controller_knu_1_0_left"
                                      : "{indexcontroller}valve_controller_knu_1_0_right");
        props->SetInt32Property(container, vr::Prop_ControllerRoleHint_Int32, role);
        auto* input = vr::VRDriverInput();
        for (auto& component : components) {
            const auto error = component.button
                ? input->CreateBooleanComponent(container, component.path, &component.handle)
                : input->CreateScalarComponent(container, component.path, &component.handle,
                    vr::VRScalarType_Absolute, component.signedAxis
                        ? vr::VRScalarUnits_NormalizedTwoSided : vr::VRScalarUnits_NormalizedOneSided);
            if (error != vr::VRInputError_None) return false;
        }
        vr::VRBoneTransform_t gripLimit[eBone_Count]{};
        hand.ComputeSkeletonTransforms(role, {1, 1, 1, 1, 1}, {}, gripLimit);
        const auto error = input->CreateSkeletonComponent(container,
            left ? "/input/skeleton/left" : "/input/skeleton/right",
            left ? "/skeleton/hand/left" : "/skeleton/hand/right", "/pose/raw",
            vr::VRSkeletalTracking_Estimated, gripLimit, eBone_Count, &skeleton);
        if (error != vr::VRInputError_None) return false;
        PublishHand(std::array<float, handSize>{}.data());
        return true;
    }

    vr::TrackedDeviceIndex_t index = vr::k_unTrackedDeviceIndexInvalid;
    Clock::time_point previousStamp{};
    std::array<double, 3> previousPosition{}, linearVelocity{};
    const vr::ETrackedControllerRole role;
    Display display;
    MyHandSimulation hand;
    vr::VRInputComponentHandle_t skeleton = vr::k_ulInvalidInputComponentHandle;
    std::array<Component, 13> components{{
        {"/input/thumbstick/x", false, true}, {"/input/thumbstick/y", false, true},
        {"/input/thumbstick/touch", true}, {"/input/trigger/value"}, {"/input/trigger/click", true},
        {"/input/trigger/touch", true}, {"/input/grip/value"}, {"/input/grip/force"},
        {"/input/grip/touch", true}, {"/input/a/click", true}, {"/input/a/touch", true},
        {"/input/b/click", true}, {"/input/b/touch", true},
    }};
};

class Provider final : public vr::IServerTrackedDeviceProvider {
    vr::EVRInitError Init(vr::IVRDriverContext* context) override {
        VR_INIT_SERVER_DRIVER_CONTEXT(context);
        WSADATA wsa{};
        winsockReady = WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
        if (!winsockReady || !OpenSocket()) {
            vr::VRDriverLog()->Log("Desktop Body: cannot bind UDP 127.0.0.1:29745");
            Cleanup();
            return vr::VRInitError_Driver_Failed;
        }
        for (auto& device : devices) {
            if (!vr::VRServerDriverHost()->TrackedDeviceAdded(device.serial, device.kind, &device)) {
                vr::VRDriverLog()->Log("Desktop Body: device registration failed");
                Cleanup();
                return vr::VRInitError_Driver_Failed;
            }
        }
        vr::VRDriverLog()->Log("Desktop Body: listening on UDP 127.0.0.1:29745");
        return vr::VRInitError_None;
    }

    void Cleanup() override {
        if (receiver != INVALID_SOCKET) {
            closesocket(receiver);
            receiver = INVALID_SOCKET;
        }
        if (winsockReady) {
            WSACleanup();
            winsockReady = false;
        }
        lastPacket = {};
        VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }
    const char* const* GetInterfaceVersions() override { return vr::k_InterfaceVersions; }
    bool ShouldBlockStandbyMode() override { return false; }
    void EnterStandby() override {}
    void LeaveStandby() override {}

    void RunFrame() override {
        const auto now = Clock::now();
        if (now < nextFrame) return;
        nextFrame = now + std::chrono::milliseconds(10);
        Receive(now);
        for (size_t i = 0; i < devices.size(); ++i) {
            devices[i].PublishPose(frame.data() + i * poseSize, lastPacket, now);
        }
        const std::array<float, handSize * 2> neutral{};
        const float* hands = now - lastPacket < inputTimeout && lastPacket != Clock::time_point{}
            ? frame.data() + poseCount * poseSize : neutral.data();
        devices[1].PublishHand(hands);
        devices[2].PublishHand(hands + handSize);
        vr::VREvent_t event{};
        while (vr::VRServerDriverHost()->PollNextEvent(&event, sizeof(event))) {}
    }

    bool OpenSocket() {
        receiver = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (receiver == INVALID_SOCKET) return false;
        const BOOL exclusive = TRUE;
        if (setsockopt(receiver, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                       reinterpret_cast<const char*>(&exclusive), sizeof(exclusive)) != 0) return false;
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        u_long nonblocking = 1;
        return ioctlsocket(receiver, FIONBIO, &nonblocking) == 0 &&
            bind(receiver, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0;
    }

    void Receive(Clock::time_point now) {
        // Bound the work done in vrserver's frame loop, including malformed datagrams.
        for (int i = 0; i < 32; ++i) {
            char packet[packetSize + 1];
            const int count = recv(receiver, packet, sizeof(packet), 0);
            if (count == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK) break;
            if (count != packetSize || memcmp(packet, "DB02", 4) != 0) continue;
            std::array<float, payloadSize> candidate;
            memcpy(candidate.data(), packet + 4, payloadSize * sizeof(float));
            if (!Valid(candidate)) continue;
            frame = candidate;
            lastPacket = now;
        }
    }

    static bool Valid(const std::array<float, payloadSize>& values) {
        for (float value : values) {
            if (!std::isfinite(value) || std::abs(value) > 10.f) return false;
        }
        for (size_t i = 0; i < poseCount; ++i) {
            const float* q = values.data() + i * poseSize + 3;
            const float normSquared = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
            if (normSquared < 0.99f || normSquared > 1.01f) return false;
        }
        for (size_t hand = 0; hand < 2; ++hand) {
            const float* controls = values.data() + poseCount * poseSize + hand * handSize;
            for (size_t i = 0; i < handSize; ++i) {
                if (controls[i] < (i < 2 ? -1.f : 0.f) || controls[i] > 1.f) return false;
            }
        }
        return true;
    }

    SOCKET receiver = INVALID_SOCKET;
    bool winsockReady = false;
    Clock::time_point lastPacket{}, nextFrame{};
    std::array<float, payloadSize> frame{
        0, 1.65f, 0, 1, 0, 0, 0,
        -0.32f, 1.15f, -0.30f, 1, 0, 0, 0,
         0.32f, 1.15f, -0.30f, 1, 0, 0, 0,
        0, 0.95f, 0, 1, 0, 0, 0,
        -0.13f, 0.06f, 0, 1, 0, 0, 0,
         0.13f, 0.06f, 0, 1, 0, 0, 0,
        0, 1.38f, 0, 1, 0, 0, 0,
        -.42f, 1.38f, 0, 1, 0, 0, 0,
         .42f, 1.38f, 0, 1, 0, 0, 0,
        -.13f, .52f, 0, 1, 0, 0, 0,
         .13f, .52f, 0, 1, 0, 0, 0,
    };
    std::array<Device, poseCount> devices{{
        {"DB-HMD", vr::TrackedDeviceClass_HMD},
        {"DB-LEFT-HAND", vr::TrackedDeviceClass_Controller, vr::TrackedControllerRole_LeftHand},
        {"DB-RIGHT-HAND", vr::TrackedDeviceClass_Controller, vr::TrackedControllerRole_RightHand},
        {"DB-WAIST", vr::TrackedDeviceClass_GenericTracker},
        {"DB-LEFT-FOOT", vr::TrackedDeviceClass_GenericTracker},
        {"DB-RIGHT-FOOT", vr::TrackedDeviceClass_GenericTracker},
        {"DB-CHEST", vr::TrackedDeviceClass_GenericTracker},
        {"DB-LEFT-ELBOW", vr::TrackedDeviceClass_GenericTracker},
        {"DB-RIGHT-ELBOW", vr::TrackedDeviceClass_GenericTracker},
        {"DB-LEFT-KNEE", vr::TrackedDeviceClass_GenericTracker},
        {"DB-RIGHT-KNEE", vr::TrackedDeviceClass_GenericTracker},
    }};
};

Provider provider;
}  // namespace

extern "C" __declspec(dllexport) void* HmdDriverFactory(const char* name, int* code) {
    if (strcmp(name, vr::IServerTrackedDeviceProvider_Version) == 0) {
        return static_cast<vr::IServerTrackedDeviceProvider*>(&provider);
    }
    if (code) *code = vr::VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
