"""Reusable prompt prefixes for bounded VRChat objectives."""

from typing import Literal


PresetName = Literal["pick_up_and_confirm", "carry_through_waypoint", "place_and_verify"]

PRESETS: dict[PresetName, tuple[str, str]] = {
    "pick_up_and_confirm": (
        "Use the free hand and locate the requested prop. Keep any occupied grip closed. "
        "Use the supplied color presets for hue and ROI. ROI coordinates are decimal fractions of image size, "
        "for example [0.2, 0.3, 0.7, 0.8]. "
        "Queue aim, outline check, grip, and a short lift in that order. For two pickups, "
        "use paired aim only when both grips are open and check each outline separately. "
        "A closed grip reports input state. After lifting, check that each requested prop follows its hand. "
        "If tracking HUD is obscured, the next batch must only clear the obstruction with an empty wrist "
        "movement or a small look. Aim again after the fresh frame. "
        "If depth works but the beam cannot reach, change the approach or aim point. "
        "If an outline check fails, change the aim and recheck before closing. Preserve the other grip on recovery. "
        "Treat missing, occupied, and unreachable props as separate outcomes. Use blocked when the available "
        "controls cannot recover within this objective.",
        "Verify only the named pickup and its hand. A glow is evidence of aim. "
        "A prop visibly following a raised hand is evidence of pickup.",
    ),
    "carry_through_waypoint": (
        "Use the supplied route and waypoint constraints. Preserve occupied grips. Batch the clear "
        "crossing into the shortest allowed route sequence. Ask for a fresh arrival check before handing "
        "off to placement. If the pose is short and the way is clear, make one small correction. "
        "If a fixture blocks the crossing, stop forward motion and find the opening before trying again. "
        "Escalate on an unrecognized route or missing held prop.",
        "Verify the requested landmark or destination from the supplied frame. "
        "A distant view of the destination alone does not establish arrival.",
    ),
    "place_and_verify": (
        "Keep the grip closed while positioning the held prop above the named support, with room "
        "inside the edge. Lower the prop before opening the grip. Leave clearance between props. "
        "If another prop is between the hand and the landing, lift clear before moving across it. "
        "Lower only after the held prop is horizontally over its landing patch. "
        "Image points and color ROI values for placement are fractions between 0 and 1. "
        "Reuse the supplied target hue and consult failed arguments before retrying. "
        "For a prop that rolls, use upright_held_object with visible top and base points when the control "
        "is available. Then reobserve before aligning the base over the support. "
        "When using align_held_surface followed by a release, supply a tolerance_m so the harness "
        "can check the remaining error and keep the grip closed if alignment fails. "
        "Use support_reference when the objective names a verified resting location for this prop "
        "and the spot is clear. Otherwise supply a visible image support point. "
        "Set premeasure_support when a wrist move may hide the chosen support point. For an upright "
        "prop, near_base measures a lower portion of the color region before lowering. "
        "Allow a reach sidestep only when the brief lateral path is visibly clear. "
        "During recovery, keep the next action batch short. Preserve the occupied grip until a landing is verified. "
        "Verify the named prop is on the support after release with the empty hand withdrawn. "
        "A similar colored prop elsewhere is not a match. If the item slips or lands outside the "
        "support, reacquire it if visible and reachable. Escalate if it leaves the permitted search area.",
        "Verify the named prop rests on the named support. Check its shape, color, and location. "
        "A visually similar prop or an item still in the hand does not count.",
    ),
}
