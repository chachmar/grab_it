import sys

sys.path.insert(0, r"C:\Users\chach\Projects\grab_it-sim")
from px150_ik import fk_xyz, ik_position

base = fk_xyz(-0.207, -1.60, 1.40)
print("tip", [round(v, 3) for v in base])
seed = {
    "waist": -0.207,
    "shoulder": -1.60,
    "wrist_angle": 1.40,
    "wrist_rotate": 0.0,
}
for dz in (0.03, 0.04, 0.05, 0.06, 0.08):
    found = False
    for pitch in (None, 0.0, -0.4, 0.4, -0.8, 0.8):
        sol = ik_position(
            base[0],
            base[1],
            base[2] + dz,
            0.0,
            seed=seed,
            avoid_create3=False,
            pitch_pref=pitch,
        )
        if sol is None:
            continue
        p = fk_xyz(sol["waist"], sol["shoulder"], sol["wrist_angle"])
        print(
            "dz",
            dz,
            "pitch",
            pitch,
            "sh",
            round(sol["shoulder"], 3),
            "wa",
            round(sol["wrist_angle"], 3),
            "got_dz",
            round(p[2] - base[2], 3),
            "got_dx",
            round(p[0] - base[0], 3),
        )
        found = True
        break
    if not found:
        print("dz", dz, "none")
