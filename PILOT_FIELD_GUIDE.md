# VRChat pilot field guide

## Create an invite-only instance

1. Open Quick Menu, select **Worlds**, then **Search Worlds**.
2. Once the text field has focus, type the world name with the desktop keyboard and press Enter. For this run the query was **The Black Cat**. The world details showed **spookyghostboo** as creator.
3. Select **New Instance**, then **Invite**. Check that the creation confirmation displays the requested world and **Invite** before pressing **Join**.
4. Wait for the destination to render. The Black Cat entrance has double doors and a reception desk with a cat emblem.

## Aim reliably

Use the headset image for selecting controls. Move the pointer, observe its visible tip and hover tooltip, then correct its aim. Menus can shift after loading, and idle motion can move a small target between frames. A tooltip naming the intended action is stronger evidence than the requested pointer coordinates. On the instance screen, the pointer initially landed on **Preload World** beside **New Instance**.

## Set the overhead view before walking

Open the **Camera** tab and select **Stream Camera** in VRChat to spawn the handheld camera. Turn on **Look At Me**. Grip the camera handle with a virtual hand, lift it above and behind the avatar, then release and return the hand to neutral. Check the actual camera feed before walking. The successful placement at this spawn used a left controller position of about (-0.25, 2.65, 0.48) meters in standing space and a Zoom setting of 20. A higher placement clipped through the entry architecture. These numbers depend on the rig, heading, and world.

## Navigate by landmarks

Walk through the front doors toward reception, turn right along the lobby, then left into the room with the stage and stairs. The circular bar wraps around the pillar opposite the stage. Stop outside the counter near the stools. Use a bounded movement pulse that always releases both axes and inspect the next frame before choosing the next segment. Timings depend on speed and starting position.

## Review a gesture

Capture the stream camera from before takeoff through the return to standing. The solver can finish its tracker clip even when VRChat's avatar IK only partially reproduces the motion. For a backflip, look for the feet passing above the head and a return to standing. A side-on fixed camera makes that easier to judge than a lens tracking the head from above.
