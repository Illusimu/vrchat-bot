# Desktop Body menu benchmarks

The Desktop Body harness should let an operator ask for an in-game outcome in one call and get back a verified result. The pilot can keep viewing frames and moving the virtual controllers locally while the caller receives a compact record of the action and its result. These benchmarks cover three workflows in VRChat. They use the headset feed and virtual controllers for navigation.

## Baseline on November 20, 2026

One live run per workflow on the same Windows and SteamVR session. Wall times are rounded estimates from the first action through the observed end state, including tool and screenshot review time. The workflows begin from different states, listed below. Headset frames count full images opened for review. Internal capture and model token use were not measured.

| Workflow and starting state | Wall time | Headset frames reviewed | Result |
| --- | ---: | ---: | --- |
| Join **The Black Cat** from Butterfly Home with menus closed | ~2m 35s | 12 | VRChat loaded Invite instance **#26258**. The creation dialog showed **Invite**, and the client log and scene-ready state confirmed the destination. Join to scene-ready took **11.8s**. |
| Change **UI/Menu** volume from The Black Cat entrance with menus closed | ~5m 22s | 25 | The Audio panel showed **0% → 25%**. Master remained at **0%**. |
| Calibrate FBT from the Audio page | ~1m 22s | 7 | The calibration view showed controller outlines. The T-pose and dual-trigger sequence ran in **3.4s**, and the outlines cleared. Avatar hip and foot tracking were not checked from a mirror or camera. |

### Join an Invite instance

Quick Menu → Worlds → Search Worlds → enter **The Black Cat** → select the result by **spookyghostboo** → New Instance → Invite → verify the created instance → Join → wait for the requested world and instance to load. Text entry used VRChat's in-game Paste button with a temporary text clipboard value. The entire search and join took place inside VRChat.

The Quick Menu initially sat low in the frame, and image-based aiming exhausted all eight probes in about 10s. Repositioning the view brought the Worlds tile into reach. Reusing a known controller pose put the search pointer about one button edge off, so the next click reached Popular instead of Paste. Verifying that the search dialog has opened before entering text would have prevented that detour. The search returned several similar names, so world title and creator both matter.

### Set UI/Menu volume

Quick Menu → Audio → Audio Volume → UI/Menu slider → read the percentage again. A head turn exposed more of the bottom tab row. The audio icon looked aligned in several frames but only responded after the visible hover tooltip appeared. The slider required a fresh read of the correct row after clicking. Trigger acknowledgments and laser intersections alone gave false confidence about both controls.

### Calibrate full-body tracking

From Audio, close and reopen Quick Menu to return to Launch Pad → Calibrate → verify the calibration view → hold T-pose and confirm with both virtual triggers → check that calibration mode exits. Closing and reopening Quick Menu reached Launch Pad after trying the small tab row. The pulse routine already handles the pose and both triggers locally. Add one bounded tracker movement in view of a mirror or stream camera to verify that the avatar follows it. Until then, report the calibration UI outcome and tracking quality separately.

## Implemented first route

`set_ui_volume(percent)` runs inside the pilot. It finds Launch Pad or Audio in the headset feed, opens Quick Menu if needed, aims at the Audio tab, and checks for the tab's hover highlight before clicking. It then requires an Audio panel match before aiming at the UI/Menu slider. It reports an estimated percentage from the visible slider fill. The tool returns the stopped step, frame ID, elapsed time, probe count, and whether it sent a click.

The panel matcher uses OpenCV feature matching and a perspective transform. Each reference covers stable parts of a panel. The transform maps known control locations from that reference to the current headset frame. This is the local similarity check. It can tolerate a tilted or scaled menu and rotating banners. The laser endpoint is measured from the headset image. An angle in tracker space alone can't confirm a UI hit. The workflow waits for fresh frames after moving and uses the observed endpoint to correct the next aim.

The pilot suppresses its breathing and weight-shift output during this route and presents the menu with a steadier left-wrist pose. It checks the projected control and the visible laser on consecutive frames before clicking. The `menu_geometry()` diagnostic matches the same panel in a simultaneous pair of eye images. For this driver, depth is **0.064 m × 320 px ÷ horizontal disparity**, with a five-pixel vertical agreement check. It returns the target in headset coordinates and an estimated panel tilt. A live Launch Pad check measured **77.8°** before raising and rotating the wrist and **7.8°** afterward. Headset geometry is diagnostic so far. The route still confirms alignment with the rendered pointer.

A live rerun from closed menus in The Black Cat requested **55%**. It opened Quick Menu and Audio, used five aim probes, and verified a change from an estimated **37%** to **51%** in **16.55s**. It returned one structured result. A headset frame was reviewed afterward for the benchmark. Earlier runs stopped when a panel or the resulting value could not be confirmed. This is one successful end-to-end rerun.

## Proposed harness changes

For the next routes, represent each step as a source panel, target control, and expected result. The result can be another panel, a value, or a verified destination. Let the caller choose the route and let the pilot execute it up to the first unexpected state. Keep the route separate from the panel references so a single updated panel can repair every route that uses it. World interaction can reuse the observe, aim, act, and verify loop. A moving object also needs a world-space pose and an interaction check, such as whether a grip actually moved the object.

To use stereo for the first aim, measure the controller's visible beam in both eyes across several known panel points and distances. Fit a ray in controller coordinates, then intersect that ray with the measured panel plane. Use the intersection to seed the existing feedback loop and keep the final hit and outcome checks. For unknown world objects, consider adding a local detector for identity. Known menu panels already have sufficient visual features for the current matcher.

1. **Identify the panel and its actionable controls.** Use stable cues such as the header, selected tab, button label or icon, and a small panel crop. Track the panel's corners as the head moves and express targets relative to the panel. Start with Launch Pad, Worlds, search input and results, the instance dialog, Audio, and calibration. Normalize the panel before OCR or template matching. Keep the capture frame ID and pose alongside every detection.
2. **Aim, confirm, click once, and verify in one local operation.** Reuse laser calibration while the panel and head pose are stable. Prefer a detection tied to the right pointer and distinguish the tip marker from a line crossing or left-hand ray. When the cursor is hidden or at its reach limit, recenter the view before spending more probes. Use hover evidence where available, then wait for the expected panel, selected row, or numeric readback. Report an uncertain click outcome without blindly repeating it. Serialize controller actions and release the trigger on timeout or cancellation.
3. **Add small workflows that build on the same operation.** A private world workflow checks title and creator, enters text only in a verified search dialog, verifies Invite privacy before joining, and waits for the world and new instance in the client state. A volume workflow takes a named channel and desired percentage, checks the slider row and reads back its value. FBT verifies calibration mode, uses the existing pose routine, and verifies avatar response when a suitable view is available.
4. **Keep routine video and retries local.** Return status, verified value or destination, duration, failed step if any, and relevant frame IDs as text. Surface a cropped diagnostic image when the outcome is ambiguous, and keep full headset images available for debugging. This lets us lower image traffic without reducing the evidence needed to verify a state change.

## How to measure the next pass

### Mapped Invite join

One live call to `join_invite_world("Butterfly Home")` started on the Audio panel and joined a new Invite instance, **#59688**, in **33.03s**. The route spent **21.67s** on menus and **11.36s** from Join to the VRChat scene-ready event. It used nine aim probes. The previous manually guided Butterfly Home run took about two minutes, including roughly the same loading time. These are single runs from different starting conditions.

During development, the route stopped at mismatched or unreachable controls and saved the current frame. A later test resumed at a verified Invite confirmation and joined that existing instance without creating another. The final run used the updated map from Audio through Join. During automated menu actions, the pilot holds the headset and controllers steady between explicit pose changes and suppresses locomotion, idle clips, and local input motion.

Run each workflow at least five times. Include the starting panel and pose in each record and test the quick menu at more than one angle. Measure wall time, client load time, time per step, number of local probes, number of calls and images surfaced to the caller, and whether the result was independently verified. Record model tokens if the runner exposes actual usage. Use median time and success count for comparison with a repeat run of the current harness.

Initial targets are one caller command per workflow, at most two images surfaced on a successful run, and no unintended clicks during the test set. Target under 45s for the private join workflow excluding client load, under 20s for volume with verified readback, and under 30s for FBT once the avatar is visible for a tracking check. These are targets for the next iteration.
