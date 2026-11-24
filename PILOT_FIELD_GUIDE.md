# VRChat pilot field guide

See [Desktop Body menu benchmarks](VRCHAT_MENU_BENCHMARKS.md) for live timings, verification criteria, and the harness design.

## Create an invite-only instance

The pilot exposes `join_invite_world("Butterfly Home")`. The game mirror also has a map for `join_invite_world("The Black Cat")`, checked against saved frames. Start from Audio, Launch Pad, Worlds, or a verified checkpoint in the search and instance flow. The [route map](menu_templates/world_join.json) includes reference images for the target card, creator line, and Invite confirmation.

If the route stops, read `step`, `expected`, `trigger_sent_at_step`, and the diagnostic frame. For a moved control, use the diagnostic to update its search region or panel-relative coordinates. For a changed screen, capture a clean crop of its stable label and, if the control changes appearance on hover, a second crop of that state. Check the expected next screen with a deliberate in-game action before saving a changed edge. The next call reloads the saved map and crops. A confirmed Invite dialog for the requested world can resume at Join, so recovery does not need to create another instance.

From a low Launch Pad, the full Worlds panel can open above the game mirror. The route checks that the search row is visible inside the frame after opening Worlds. If needed, it looks upward in at most two steps and checks the same panel before issuing another UI action. A live check recovered the search row from a clipped view with two head turns and no trigger presses.

1. Open Quick Menu, select **Worlds**, then **Search Worlds**.
2. Confirm the search input dialog opened before entering the world name. VRChat's in-game **Paste** button worked in the headset run. Submit the query, then verify the world title and creator. **The Black Cat** is by **spookyghostboo**.
3. Select **New Instance**, then **Invite**. Check that the creation confirmation displays the requested world and **Invite** before pressing **Join**.
4. Wait for the destination to render. The Black Cat entrance has double doors and a reception desk with a cat emblem.

## Aim reliably

Use the headset image for selecting controls. Move the pointer, observe its visible tip and hover tooltip, then correct its aim. Menus can shift after loading, and idle motion can move a small target between frames. A tooltip naming the intended action is stronger evidence than the requested pointer coordinates. On the instance screen, the pointer initially landed on **Preload World** beside **New Instance**.

## Set the overhead view before walking

With `--game-mirror`, install `SpoutGL==0.1.1` in the pilot's Python environment and enable the **OSC** toggle in VRChat's main Settings. Open **Camera → Photo Camera** with the virtual controllers, then call `camera(enabled=True)`. Use `observe(view="headset")` for first-person controls and `observe(view="camera")` for the independent Spout frame. Disconnect with `camera(enabled=False)` when finished. [VRChat Spout behavior](https://docs.vrchat.com/docs/vrchat-202433-openbeta)

With native SteamVR capture, open **Stream Camera** and connect the pilot as before. Turn on **Look At Me** for an overhead follow shot. Grip the camera handle with a virtual hand, lift it above and behind the avatar, then release and return the hand to neutral. Check the actual camera feed before walking. The successful placement at this spawn used a left controller position of about (-0.25, 2.65, 0.48) meters in standing space and a Zoom setting of 20. A higher placement clipped through the entry architecture. These numbers depend on the rig, heading, and world.

## Navigate by landmarks

Walk through the front doors toward reception, turn right along the lobby, then left into the room with the stage and stairs. The circular bar wraps around the pillar opposite the stage. Stop outside the counter near the stools. Use a bounded movement pulse that always releases both axes and inspect the next frame before choosing the next segment. Timings depend on speed and starting position.

## Review a gesture

Capture the stream camera from before takeoff through the return to standing. The solver can finish its tracker clip even when VRChat's avatar IK only partially reproduces the motion. For a backflip, look for the feet passing above the head and a return to standing. A side-on fixed camera makes that easier to judge than a lens tracking the head from above.

`gesture` accepts `repetitions` from 1 through 5. Push-ups use one continuous set with the requested number of down-up cycles. The tool returns the commanded count, and camera review verifies what the avatar actually did.
