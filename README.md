# Desktop Body for SteamVR

Desktop Body turns keyboard and mouse input into a simulated VR headset, two Index-style controllers, and eight virtual body trackers. The intended setup is a person playing VRChat in SteamVR mode from their monitor. A local animation layer can add gestures while the player controls locomotion and the view.

The first build includes a desktop pilot window and a Windows x64 OpenVR driver. The driver package is staged under `build/package/desktopbody`. Live headset rendering, VRChat input bindings, and full-body calibration are pending a dedicated test session.

## How tracking works

SlimeVR uses headset position, measured rotations, and body proportions to compute virtual tracker poses. Its OpenVR driver exposes those poses to SteamVR. Desktop Body follows that separation. The desktop pilot generates motion, a single rig calculates eleven poses relative to a shared floor and heading, and the SteamVR driver publishes them. The current rig uses procedural steps and crouches. The waist and feet are ready for a more detailed body solver with foot planting. [SlimeVR 101](https://docs.slimevr.dev/slimevr101.html) · [SlimeVR OpenVR driver](https://github.com/SlimeVR/SlimeVR-OpenVR-Driver)

The two hands use SteamVR's `knuckles` profile for Index bindings. Valve's sample hand solver converts five independent curls per hand into 31-bone poses for both skeletal motion ranges. The headset is a virtual display with a standing tracking space. It supplies the app's camera pose, including a simulated crouch. [Valve driver API](https://github.com/ValveSoftware/openvr/blob/master/docs/Driver_API_Documentation.md) · [Valve hand simulation sample](https://github.com/ValveSoftware/openvr/tree/master/samples/drivers/drivers/handskeletonsimulation)

`Body.request_gesture(name, now)` is the interface for an eventual AI assistant. Wave and nod are the initial finite gestures. Direct look input cancels a nod, and direct right-hand input cancels a wave. The left stick always comes from the player. Escape and focus loss release the controls. The driver also zeros both controllers' input values and curls after 250 ms without a valid frame, while keeping the last tracked positions.

## Build

Use Python 3 with Tk, CMake, and Visual Studio Build Tools with Desktop development with C++. Extract [OpenVR SDK v2.12.14](https://github.com/ValveSoftware/openvr/releases/tag/v2.12.14) to `build/deps/openvr-2.12.14` or point CMake to an existing copy.

```powershell
cmake -S . -B build/native -G "Visual Studio 17 2022" -A x64 -DOPENVR_SDK=build/deps/openvr-2.12.14
cmake --build build/native --config Release
cmake --install build/native --config Release --prefix build/package
```

## First SteamVR session

Use these steps after finishing your current VR session.

1. Exit SteamVR and VRChat. Disconnect the physical headset for this trial. Locate the SteamVR installation and the config directory used by it.
2. From the workspace, register the staged driver. Adjust the first path if SteamVR is installed in another library.

   ```powershell
   $steamvr = "C:\Program Files (x86)\Steam\steamapps\common\SteamVR"
   $driver = (Resolve-Path "build/package/desktopbody").Path
   & "$steamvr\bin\win64\vrpathreg.exe" adddriver $driver
   ```

3. Back up your `steamvr.vrsettings`. Merge the entries below into it, keeping the existing keys in those sections. The usual location is the Steam installation's `config` folder.

   ```json
   {
     "steamvr": {
       "forcedDriver": "desktopbody"
     },
     "driver_desktopbody": {
       "enable": true
     }
   }
   ```

4. Start SteamVR. Check for the virtual HMD, left and right controllers, and eight trackers. In **Settings → Controllers → Manage Trackers**, check waist, chest, both feet, both knees and both elbows against their matching DB serials.
5. Run `python pilot.py` and click the preview to arm input. Put the pilot beside SteamVR's **Display VR View**. Move the view and check that the mirror follows it. The pilot preview shows the rig's outgoing poses and finger counts.
6. Launch VRChat in VR mode. Check movement, trigger/grip bindings, and finger animation on an avatar that supports them. Enter full-body calibration and hold **T** to produce the calibration pose. While holding **T**, use **Z** and **X** to send both triggers. [VRChat full-body calibration](https://docs.vrchat.com/docs/full-body-tracking)

To return to normal VR, exit SteamVR, restore the settings backup, and run `& "$steamvr\bin\win64\vrpathreg.exe" removedriver $driver`. [Valve driver registration](https://github.com/ValveSoftware/openvr/blob/master/docs/Driver_API_Documentation.md#building--development-environment)

## Pilot controls

| Control | Input |
| --- | --- |
| Left thumbstick | WASD |
| Head orientation | Arrow keys or right mouse drag |
| Crouch / calibration pose | C / hold T |
| Left and right grips | Q / E |
| Left and right triggers | Z / X |
| Left A and B | G / V |
| Right A and B | Space / B |
| Right fingers from thumb to pinky | 1–5 |
| Left fingers from thumb to pinky | 6, 7, 8, 9, 0 |
| Assisted gesture | F to wave / N to nod |
| Pause | Escape or leave the pilot window |

## Pose stream

The pilot sends one 400-byte UDP datagram per update to `127.0.0.1:29745`. A four-byte ASCII version marker, `DB02`, is followed by 99 little-endian float32 values.

- Eleven poses in headset, left hand, right hand, waist, left foot, right foot, chest, left upper arm, right upper arm, left knee, right knee order. Each has position `x, y, z` in meters and a unit quaternion `w, x, y, z`. The coordinate system is OpenVR standing space, with Y up, negative Z forward, and the floor at Y = 0.
- Eleven controls per hand, left then right. The order is stick X/Y in `[-1, 1]`, trigger, grip, A, B, then five finger curls in thumb-to-pinky order. Trigger, grip, buttons, and curls use `[0, 1]`.

## Run a local visual objective

Start the virtual driver and VRChat as above. In one PowerShell window run
`./start-vr-model.ps1`. In another run `python pilot.py`, enter an objective,
and press **Run**. **Stop** cancels the objective. Escape works while the preview
has focus. Manual controls release when the preview loses focus.
Keep only one pilot open so two processes cannot compete for the driver.

The launcher uses the workspace copies of llama.cpp CUDA 12.4, the Q4_K_M MTP
GGUF and F16 vision projector, and selects the V100S by UUID. The server listens
on loopback port 8097. The pilot captures the left eye of the virtual headset
window through Windows Graphics Capture, including dashboard overlays. When that
window is absent it uses the VRChat window. A blank frame triggers a compositor read.
The model sends a target to a local solver. For navigation, the model marks a
visible object and specifies its desired horizontal position and apparent size.
The solver tracks that patch across fresh frames, issues short joystick pulses,
and reports arrival, loss of tracking, or timeout. Apparent size is a stand-in
for distance. A change in viewpoint or a similar-looking object can confuse
the tracker. The model sees the next frame and solver outcome before replanning.

A wrist target specifies a side, position in meters (right, height above floor,
negative forward), and yaw, pitch and roll in degrees, relative to the heading
at the start of the reach. The solver generates a smooth trajectory from the
current tracked pose, then applies the requested grip or trigger and holds the
endpoint. Stop releases held targets. VRChat handles the avatar elbow and arm
IK. The pilot can verify the commanded wrist pose; pickup or click success
requires visual confirmation. Head orientation and the existing wave, nod and
backflip clips also have completion outcomes.

## Generated motion and camera endpoints

For a whole-body action beyond wave, nod, or backflip, the planner names a clip
and supplies a duration and sparse SOMA joint or hip keyframes. The solver
runs the installed Kimodo worker on the V100S and caches successful clips in
`build/motions/generated`. Repeated names reuse the clip. Playback aligns the
hips and heading to the rig, blends in and out over 250 ms, and resamples all
eleven tracker poses at the pilot update rate. Grip and finger inputs stay under
the controller system. Stop or manual takeover cancels pending generation.
The upper-arm and knee targets follow points along the generated upper arms and thighs and carry those segments� rotations. Built-in exercises solve these targets from their endpoints. After opening Calibrate FBT, an MCP pilot can call `confirm_fbt` and inspect the avatar from the stream camera. The worker uses the local checkpoint and keyframe conditioning.

Enable OSC in VRChat and select **VRChat camera** in the pilot for a second view.
The pilot requests Stream mode with Spout off, captures the VRChat desktop
window as the camera, and obtains the headset image independently. It restores
the prior camera mode and Spout setting when disconnected. Camera movement
requires VRC+ Camera Dolly, default OSC ports 9000/9001, and time-based dolly
playback. The pilot imports a new path for each shot, replacing the active
dolly path. Export a path first if you want to keep it.

The planner specifies only the desired player-local camera position and Unity
rotation. For example, a lens three meters in front of the avatar, facing back
at it, is `[0,1.4,3,5,180,0]` in right/up/forward meters and pitch/yaw/roll
degrees. The first shot establishes a reference. Subsequent paths move from
the last held endpoint using a duration based on distance and angular change.
The solver waits for a playback acknowledgment and holds the shot for up to
60 seconds. The model checks framing against the next camera image and can
request another endpoint. During a gesture the pilot also retains a camera
frame near the middle for the next decision. Playback acknowledgment does
not measure absolute lens position or obstacle clearance.

[VRChat Camera Dolly and OSC](https://wiki.vrchat.com/wiki/Camera_Dolly)

## Menu control over MCP

Launch a single pilot with `python pilot.py --mcp`. The streamable HTTP endpoint
is `http://127.0.0.1:29746/mcp`. Add `--camera` to connect the Stream camera
at launch, or call the MCP camera tool later. Tools are available to local
processes while this option is active.

Use `observe(view="headset")`, then `menu(action="toggle")` to open the Quick
Menu with the left Index B button. Observe again and aim with, for example,
`menu(action="aim", x=0.6, y=0.4)`. Coordinates are fractions of the headset
image from its top left. Check which item highlights, adjust aim as needed,
and use `menu(action="click")` to pulse the right trigger. Each menu call waits
for a subsequent headset frame. Inspect it to confirm the visible change.
Custom controller bindings and menu depth can affect aiming and selection.

Use `run_objective` for the local visual planner, `status` for progress, and
`stop` to release inputs. Direct menu actions share the same solver as the
local planner. Stop the active objective before switching to direct control.

## Named menu controls

Both the local planner and MCP can call `select_by_name(label="New Instance")`.
The example opens the Create New Instance dialog from a visible New Instance button.
Selection holds the current tracker poses while moving the right pointer, reads
the laser and hover tooltip twice, sends one trigger pulse, and checks for the
new heading on two subsequent frames. The English Here menu has small visual
references to supplement Tesseract OCR. Other visible controls can supply
`label`, `tooltip` and `expected` (a distinctive new screen heading). Selection
requires Tesseract on PATH. For an unverified click, inspect the current
headset view before retrying.

The MCP `world_context()` tool and the local planner read the active client's
confirmed room join from its output log. They receive the world ID, display
name and an opaque instance key. During a transition or when logging is
unavailable the reader can report unknown. World-space coordinates require a
separate source.
