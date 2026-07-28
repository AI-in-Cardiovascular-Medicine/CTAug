# CTAug


**CTAug** is an open-source Python library of cardiac CT (CCT)–specific data augmentations, developed to improve the robustness of segmentation models to clinical image degradation. It was introduced as part of a unified framework for comprehensive cardiac CT segmentation and phenotyping (Mohammadi Kazaj et al., 2026).

> A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping: Human-in-the-Loop Data Annotation, Vision Foundation Model Development, Multicenter Evaluation and Clinical Validation
> [arXiv:2607.11287](https://arxiv.org/abs/2607.11287)

## Overview

Clinical cardiac CT scans frequently contain artifacts — metallic implants, pacemaker wires, calcifications, and step-and-shoot acquisition motion — that are underrepresented in curated training datasets but common in routine practice. CTAug simulates these artifact categories during training to close that gap, without requiring additional annotated data.

Across five external test datasets, training segmentation models with CTAug increased mean Dice score from 95.5 to 96.3 (Wilcoxon signed-rank test, p < 0.001 across all comparisons), with the largest benefit observed on the most artifact-rich cohort (94.81 vs. 92.45 Dice), where it raised the lower tail of the Dice distribution and narrowed its spread.

![CTAug examples and effect on segmentation robustness](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/sfigu_ctaug.png)
*(a) Representative original vs. augmented image pairs for each simulated artifact category; red arrows indicate the introduced artifacts. (b) Dice score across all structures on the five external test datasets for models trained without (gray) vs. with (red) CTAug. Source: Supplementary Figure S1, [arXiv:2607.11287](https://arxiv.org/abs/2607.11287).*

## Augmentations

| Augmentation | Description | Default probability |
|---|---|---|
| Calcification | Simulated as small, high-HU deposits | 15% |
| Wire | Linear or curvilinear high-HU structures (e.g., pacemaker leads) | 15% |
| Metal | High-HU regions with radiating bright and dark streak artifacts | 10% |
| Step-and-shoot | Stepwise intensity discontinuities with 1–2.5% spatial shifts, simulating gated acquisition motion | 10% |

For metal, wire, and calcification augmentations, placement is guided by the segmentation mask to ensure anatomically plausible localization. The step-and-shoot row corresponds to `StepMotionTransform`, which applies both the intensity step and the spatial shift; the paper calls it "step" for simplicity.

## Installation

```bash
pip install ctaug          # core (torch-based transforms only)
pip install ctaug[monai]     # + MONAI dictionary-transform adapters
```
if repo is cloned:
```bash
cd ctaug
pip install .          # core (torch-based transforms only)
pip install .[monai]     # + MONAI dictionary-transform adapters
```

Core installs `numpy`, `torch`, `scipy`, and `scikit-image`. The `monai` extra additionally installs `monai`, and unlocks `ctaug.monai`.

## Usage

### Core (plain torch)

Every transform is a dict-in/dict-out callable: it reads `data["image"]` (a `(Z, Y, X)` or `(1, Z, Y, X)` torch tensor or NumPy array), optionally `data["segmentation"]` (used to anchor artifact placement, and shifted alongside the image by the step transforms), and returns the augmented dict with the result written back plus a `<ClassName>_info` entry describing everything that was sampled. Output shape and container type (torch tensor or NumPy array) always match the input; the artifact transforms also preserve the input dtype, while the step transforms return `float32` because of their intensity rescaling.

The snippets below are the calls from [`example/example_notebook.ipynb`](https://github.com/AI-in-Cardiovascular-Medicine/CTAug/blob/main/example/example_notebook.ipynb), trimmed to the transform itself.

#### Artifact transforms — calcification, wire, metal

```python
from deep_utils import SITKUtils                 # pip install ctaug[example]
from ctaug import CalcificationTransform, MetalTransform, WireTransform

img_arr, img_itk = SITKUtils.get_array_img("10151662.nii.gz")        # (Z, Y, X)
seg_arr, _ = SITKUtils.get_array_img("10151662_seg.nii.gz")

augmentor = MetalTransform(
    spacing=img_itk.GetSpacing()[::-1],   # CTAug wants (z, y, x); SimpleITK reports (x, y, z)
    include_labels=(2, 3, 4),             # anchor placement inside these labels only
    exclude_labels=None,                  # inverse of include_labels; default (0,), i.e. skip background
    intensity=(7000, 8000),               # scalar or (low, high); default (1000, 5000)
    severity=(0.9, 0.99),                 # streak strength; default (0.1, 0.9)
    verbose=True,
)

# the notebook adds a channel axis; a plain (Z, Y, X) array works too
out = augmentor(image=img_arr[None, ...], segmentation=seg_arr[None, ...])
augmented_image = out["image"][0]
info = out["MetalTransform_info"]         # implant_specs, severity, spacing, selected_labels, ...
```

`CalcificationTransform` and `WireTransform` take the same arguments; the notebook calls all three through one helper and leaves `intensity`/`severity` at their defaults for those two. `RandomArtifactTransform` picks one of the three per call (optionally with `exclude_class=("wire",)`), so its `info` lands under the **chosen** transform's key — `MetalTransform_info`, `WireTransform_info`, or `CalcificationTransform_info` — not under its own name.

These three forward-project and reconstruct every affected slice, which is the expensive part; the notebook centre-crops to `(128, 256, 256)` first to keep the runtime sane.

#### Step-and-shoot transforms

```python
from ctaug import MotionTransform, StepMotionTransform, StepTransform

augmentor = StepMotionTransform(
    cut_off_pixel_value_weight=(0.1, 1),  # offset relative to mean intensity; default (0.2, 0.6)
    motion_move_range=(0.01, 0.1),        # fraction of the moved axis' extent; default (0.01, 0.015)
)

# no channel axis or torch conversion needed — a plain (Z, Y, X) array round-trips as-is
out = augmentor(image=img_arr, segmentation=seg_arr)
augmented_image = out["image"]
augmented_segmentation = out["segmentation"]      # shifted by the same amount
info = out["StepMotionTransform_info"]
```

These are cheap (no projection involved), so the notebook runs them on the full `(275, 512, 512)` volume. `StepTransform` takes only `cut_off_pixel_value_weight` and treats `segmentation` as optional; `MotionTransform` takes only `motion_move_range` and **requires** a segmentation, since it translates anatomy.

Every transform can be dropped into a custom augmentation pipeline (nnU-Net/batchgenerators, TorchIO, or your own training loop) the same way. The notebook also shows replaying a saved `<ClassName>_info.json` through `ctaug.metal.functional.simulate_artifacts_unified` to reproduce or hand-edit a specific implant.

### MONAI

With the `monai` extra installed, `ctaug.monai` exposes the same augmentations as standard MONAI dictionary transforms (`MapTransform` + `RandomizableTransform`), so they compose with `monai.transforms.Compose` like any other `*d` transform:

```python
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged
from ctaug.monai import Calcificationd, Metald, StepMotiond, Wired

spacing = (0.5, 0.36, 0.36)   # (Z, Y, X), in mm

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # the paper's per-artifact probabilities, applied independently
    Calcificationd(keys=["image"], spacing=spacing, label_key="label",
                   include_labels=(2, 3, 4), prob=0.15),
    Wired(keys=["image"], spacing=spacing, label_key="label",
          include_labels=(2, 3, 4), prob=0.15),
    Metald(keys=["image"], spacing=spacing, label_key="label",
           include_labels=(2, 3, 4), prob=0.10),
    StepMotiond(keys=["image"], label_key="label", prob=0.10),
])
```

`Calcificationd`, `Wired`, and `Metald` all take the same arguments (`max_n_specs`, `intensity`, `severity`, `exclude_labels`, `include_labels`, …), forwarded to the core transform. If you would rather draw *one* of the three per sample instead of rolling for each independently, swap them for a single `RandomArtifactd(keys=["image"], spacing=spacing, label_key="label", prob=0.4)`, optionally narrowed with `exclude_class=("wire",)`. `Stepd` and `Motiond` are likewise available as the two halves of `StepMotiond`.

## Example data

The example case used below is attached to the [latest release](https://github.com/AI-in-Cardiovascular-Medicine/CTAug/releases/latest) rather than committed to the repository. Download it into `example/` before running the notebook:

```bash
cd example
wget https://github.com/AI-in-Cardiovascular-Medicine/CTAug/releases/latest/download/10151662.nii.gz
wget https://github.com/AI-in-Cardiovascular-Medicine/CTAug/releases/latest/download/10151662_seg.nii.gz
```

That is one image (`10151662.nii.gz`) plus its segmentation (`10151662_seg.nii.gz`), which is what the notebook expects.

The full annotated dataset — **1020 cardiac CT cases** with segmentations — is available on [Hugging Face](https://huggingface.co/datasets/AI-CVM/Cardiac-CT), if you want to run the augmentations across more anatomy than this single scan.

## Examples on a real scan

Running the transforms on a real CCT volume ([`example/example_notebook.ipynb`](https://github.com/AI-in-Cardiovascular-Medicine/CTAug/blob/main/example/example_notebook.ipynb)) writes the augmented image next to the original and reports exactly what was sampled. All examples below come from the same scan, read with SimpleITK so the array is indexed `[z, y, x]` — note the axis-order flip against the viewer: array axis 0 is the **third** cursor coordinate in ITK-SNAP's `(x, y, z)` readout, and array axis 2 (`-1`) is the **first**. The step examples use the full `(275, 512, 512)` volume; the calcification, wire, and metal examples use the centre crop to `(128, 256, 256)` that the notebook applies to the artifact transforms.

### Calcification (`CalcificationTransform`)

```
[INFO] Augment output: example/artifact_results/10151662/aug_CalcificationTransform_img.nii.gz, info:
{'implant_specs': [{'type': '3d', 'center_mm': [18.0, 48.34375, 55.09765625], 'radius_mm': (3, 1.5782161251475086, 1.125318961588682), 'intensity': 1603},
                   {'type': '3d', 'center_mm': [16.0, 26.66015625, 73.58203125], 'radius_mm': (6, 0.8766726240357127, 0.8429775449553721), 'intensity': 4203}],
 'severity': 0.779948448472995, 'spacing': (0.5, 0.35546875, 0.35546875), 'selected_labels': [4, 2],
 'angle_size': 285, 'cls_name': 'CalcificationTransform'}
```

![Simulated calcification inserted by CalcificationTransform, inspected in ITK-SNAP](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/calcification.png)
*Axial slice (39 of 128) of `aug_CalcificationTransform_img.nii.gz` in ITK-SNAP, zoomed onto the mediastinum. The crosshair is on deposit 2; the fainter deposit 1 is visible above and to the left of it.*

Finding a deposit in the viewer takes a little arithmetic, because **`center_mm` and `radius_mm` are in millimetres, not voxels** — divide them element-wise by `spacing` (also `(z, y, x)`) to get an index you can type into the cursor box:

| | `center_mm` (z, y, x) | ÷ `spacing` → voxel (z, y, x) | ITK-SNAP cursor (x, y, z) |
|---|---|---|---|
| deposit 1 | `[18.0, 48.34, 55.10]` | `36, 136, 155` | `155, 136, 36` |
| deposit 2 | `[16.0, 26.66, 73.58]` | `32, 75, 207` | `207, 75, 32` |

So the first deposit sits at roughly **z ≈ 36** — `18.0 mm / 0.5 mm = 36`. Note the screenshot's cursor reads `(208, 76, 39)` rather than deposit 2's exact `(207, 75, 32)`: `x` and `y` land within a voxel, but the slice is off by 7, because the same `radius_mm` conversion makes this deposit 12 voxels in `z` (`6 mm / 0.5 mm`) against only ~2.4 in-plane (`0.88 mm / 0.355 mm`). It is a thin, z-elongated speck spanning slices ~20–44, so any slice in that range shows it and pinning the exact centre by eye is fiddly. Deposit 1 is squatter — 6 voxels in `z`, ~4.4 × 3.2 in-plane, spanning slices ~30–42.

The rest of the `info`:

- **`intensity: 1603` / `4203`** — the HU value written into each ellipsoid before the streak simulation. The transform clips its output back to the input volume's original range, whose maximum here is 1090 HU — which is exactly what the cursor reports, so the nominal 4203 is not measurable in the output.
- **`selected_labels: [4, 2]`** — the labels whose interiors anchored the two placements (this notebook cell passes `include_labels=(2, 3, 4)`), which is how the deposits land in plausible anatomy instead of in air.
- **`severity: 0.78`, `angle_size: 285`** — streak strength, and the number of projection angles spread over 0–180° used to forward/back-project the implant.

### Wire (`WireTransform`)

```
[INFO] Augment output: example/artifact_results/10151662/aug_WireTransform_img.nii.gz, info:
{'implant_specs': [{'type': 'wire', 'length': 23, 'center_mm': [13.5078125, 55.09765625], 'arc_radius_mm': 11,
                    'wire_radius_mm': 0.028909099449138553,
                    'angle_range': (0.10242761948965777, 2.0707460863345406),
                    'z_range_mm': [45.0, 50.0], 'intensity': 3804}],
 'severity': 0.6082850812229637, 'spacing': (0.5, 0.35546875, 0.35546875), 'selected_labels': [3],
 'angle_size': 441, 'cls_name': 'WireTransform'}
```

![Simulated pacemaker-lead wire inserted by WireTransform, inspected in ITK-SNAP](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/wire.png)
*Sagittal slice `x = 186` of `aug_WireTransform_img.nii.gz` in ITK-SNAP — the slice through the arc's apex, where the curve runs closest to in-plane and so shows the longest stretch of wire (the beaded bright streak upper right).*

`center_mm` gives only an **approximate** anchor, not a point on the wire. It has two entries, not three — the in-plane centre in millimetres — and the wire is an *arc* sweeping `arc_radius_mm` away from it:

- **`center_mm: [13.51, 55.10]`** ÷ the in-plane spacing = voxel `(38, 155)` on array axes 1 and 2, i.e. cursor `x ≈ 155`, `y ≈ 38`. That is the centre of curvature — start there, then look `arc_radius_mm` outward.
- **`arc_radius_mm: 11`** = `11 / 0.355 ≈ 31` voxels in-plane. The screenshot's slice is `x = 186`, which is exactly `155 + 31`: the apex of the arc, the far side of the circle from the anchor.
- **`z_range_mm: [45.0, 50.0]`** ÷ `spacing[0]` = slices **90–100**. The wire climbs steadily through those 11 slices as it curves, like a lead threaded through a vessel.
- **`angle_range: (0.102, 2.071)` rad** = 5.9°–118.6°, the swept portion of the circle. Combined with the radius, the traced arc spans roughly `x = 158–186` and `y = 23–69`.
- **`length: 23`** is a **point count, not a distance** — the arc is sampled at 23 positions. Those cover ~61 voxels of in-plane arc, so consecutive samples land ~2.8 voxels apart and the rasterized wire comes out beaded rather than solid, which is what the screenshot shows.
- **`wire_radius_mm: 0.029`** converts to Gaussian sigmas of `(0.06, 0.08, 0.08)` voxels — below one voxel, so it does not thicken the track at all; the wire stays one voxel wide. Raise it to fuse the beads into a continuous lead.
- **`intensity: 3804`**, **`severity: 0.61`**, **`angle_size: 441`**, **`selected_labels: [3]`** — as for the other artifact transforms; the nominal intensity is again not measurable in the output after the back-projection round trip and the clip to the input range.

### Metal (`MetalTransform`)

```
info:
{'implant_specs': [{'type': '2d', 'shape': 'ellipse', 'slices': [92, 93, 94, 95, 96, 97, 98, 99],
                    'center_px': [49, 129], 'radius_mm': [1.2, 1.1026557582413796], 'intensity': 8000},
                   {'type': '2d', 'shape': 'rectangle', 'slices': [21],
                    'center_px': [184, 154], 'radius_mm': [1.0034986741731173, 1.5974926626397938], 'intensity': 4940}],
 'severity': 0.99, 'spacing': [0.5, 0.35546875, 0.35546875], 'selected_labels': [3, 2],
 'angle_size': 454, 'cls_name': 'MetalTransform'}
```

![Simulated metal implant inserted by MetalTransform, inspected in ITK-SNAP](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/metal.png)
*Heavily zoomed axial slice of the augmented volume in ITK-SNAP, at the cursor `(x, y, z) = (130, 51, 96)` — the **first** implant, mid-way through its slice range. It is the small bright blob on the crosshair.*

The 2D implant types are **easier to locate than the 3D calcification**, because the numbers are already in index space:

- **`slices: [92, …, 99]`** — the exact z indices the implant is burned into, so no conversion at all: implant 1 covers slices 92–99 (8 of them) and implant 2 only slice **21**, a single slice. The screenshot is at `z = 96`, inside implant 1's range.
- **`center_px: [49, 129]`** — already in **pixels**, unlike `center_mm` above, and ordered `(row, column)` = (array axis 1, axis 2). The viewer swaps them, so this is cursor `x = 129`, `y = 49` — matching the screenshot's `(130, 51, …)` to within a couple of voxels.
- **`radius_mm: [1.2, 1.103]`** — the one value still in millimetres; divided by the in-plane spacing it gives **3.4 × 3.1 px**. Nearly equal radii, which is why this one reads as a compact blob. Implant 2 is `2.8 × 4.5 px` and a `rectangle` rather than an `ellipse` — the shape is drawn per implant from circle/ellipse/rectangle.

The rest of the `info`:

- **`intensity: 8000`** — burned in before the streak simulation, but not what you measure afterwards: the cursor reads 774 HU. Each affected slice is forward-projected and reconstructed by filtered back-projection (that round trip is what *creates* the streaks), which spreads the few-pixel spike out, and the result is then clipped to the input volume's range (max 1090 HU here).
- **`severity: 0.99`, `angle_size: 454`** — a very strong dampening of the sinogram rows under the implant, over 454 projection angles; `selected_labels: [3, 2]` are the labels that anchored the two placements.

Both of those are past the defaults, and both are constructor arguments that take either a scalar or a `(low, high)` range: `intensity=8000` or `intensity=(1000, 5000)` (the default), and `severity=0.99` or `severity=(0.1, 0.9)` (the default). Editing a saved `<TransformName>_info.json` and replaying it through `info_path` (see the notebook's last cell) gives the same control per implant, if you want to place one deliberately rather than sample it.

### Step artifact (`StepTransform`)

```
[INFO] img: (275, 512, 512) and seg: (275, 512, 512)
[INFO] Augment output: example/artifact_results/10151662/aug_StepTransform_img.nii.gz, info:
{'cut_off_position': 325, 'normalize': True, 'intensity_value': -135.19850158691406, 'cutoff_index': -1}
```

![Intensity step introduced by StepTransform, inspected in ITK-SNAP](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/step.png)
*Axial slice (z = 138) of `aug_StepTransform_img.nii.gz` in ITK-SNAP. The cursor is parked at (x, y, z) = (**312**, 4, 138) — deliberately a few voxels to the side of the seam, so the crosshair does not sit on the boundary and hide it.*

Reading the `info` against the image:

- **`cutoff_index: -1`** — the step is introduced along the **last array axis**, i.e. axis 2 of `(z, y, x)`. That is the viewer's `x` (right–left) axis, so in an axial slice the seam appears as a **vertical** line running top to bottom.
- **`cut_off_position: 325`** — the boundary sits at index 325 along that axis, a few voxels to the right of the crosshair at `x = 312` in the screenshot. Everything beyond it is uniformly brighter; the change is easiest to see where the line crosses the heart and the lung.
- **`intensity_value: -135.2`** — the constant offset added to that one side. The sign only says which side is darkened; which of the two halves receives it is drawn per call.
- **`normalize: True`** — after the offset, the whole volume is rescaled back to the input's min/max range, so the visible contrast across the seam is slightly smaller than the raw 135 HU offset. Pass `normalize=False` to keep the offset literal.

### Motion (`MotionTransform`)

```
[INFO] img: (275, 512, 512) and seg: (275, 512, 512)
[INFO] Augment output: example/artifact_results/10151662/aug_MotionTransform_img.nii.gz, info:
{'move_index': 0, 'cutoff_index': 2, 'cropped_value': 'zero', 'cut_off_position': 57, 'move_value': 23, 'move_left': False}
```

![Motion displacement introduced by MotionTransform, inspected in ITK-SNAP](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/motion.png)
*Axial slice (89 of 275) of `aug_MotionTransform_img.nii.gz` in ITK-SNAP, with the segmentation overlaid. Here the cursor sits **exactly on** the seam, at (x, y, z) = (**57**, 231, 89).*

Reading the `info` against the image:

- **`cutoff_index: 2`** — the volume is split along the **last array axis** again, so the seam is the vertical crosshair line, this time near the left edge of the field.
- **`cut_off_position: 57`** — the split is at index 57, matching `x = 57` under the cursor. Everything at or beyond it — nearly the whole visible slice — is the displaced block; only the thin strip to its left is left in place.
- **`move_index: 0`, `move_value: 23`, `move_left: False`** — unlike the step transform, nothing is added to the intensities: the block is *translated* by 23 voxels along the slice (Z) axis, toward increasing `z`. Because the shift is through-plane, the two sides of the seam show anatomy 23 slices apart, so structures crossing the line are broken rather than merely re-shaded.
- **`cropped_value: 'zero'`** — the 23 slices vacated at the low-`z` end of the displaced block are zero-filled, visible as a blank region if you scroll to the first slices of the volume.

The segmentation is translated by the same 23 voxels, so labels stay aligned with the displaced anatomy.

### Step + motion (`StepMotionTransform`)

> **This is the variant used in the paper.** The augmentation reported there as "step-and-shoot" is `StepMotionTransform` — the combined intensity step *and* spatial shift — shortened to just "step" in the text for simplicity. `StepTransform` and `MotionTransform` above are its two halves, exposed separately for finer-grained control.

```
[INFO] img: (275, 512, 512) and seg: (275, 512, 512)
[INFO] Augment output: example/artifact_results/10151662/aug_StepMotionTransform_img.nii.gz, info:
{'move_index': 2, 'cutoff_index': 0, 'cropped_value': 'zero', 'cut_off_position': 254, 'move_value': 40, 'move_left': False}
```

![Step-and-shoot seam introduced by StepMotionTransform, inspected in ITK-SNAP](https://raw.githubusercontent.com/AI-in-Cardiovascular-Medicine/CTAug/main/assets/step_and_motion.png)
*Sagittal view of `aug_StepMotionTransform_img.nii.gz` in ITK-SNAP, with the segmentation overlaid. The cursor sits on the seam at (x, y, z) = (217, 290, **254**).*

Reading the `info` against the image:

- **`cutoff_index: 0`** — this time the seam is introduced along the **first array axis**, i.e. the slice (Z) axis, which the viewer reports as `z`.
- **`cut_off_position: 254`** — the cut lands at index 254 of 275 along that axis, matching `z = 254` under the cursor. In the sagittal view this is the horizontal crosshair line; the slices at or above it (254–274) form the displaced block, and everything below is left untouched — visible as the offset between the vessel cross-section at the very top of the view and its continuation below the line.
- **`move_index: 2`, `move_value: 40`, `move_left: False`** — that block is displaced by 40 voxels along the last array axis (X), toward increasing X.
- **`cropped_value: 'zero'`** — within the displaced block, the 40-voxel strip vacated at the low-X edge is filled with zeros (alternatives: `'mean'`, `'cut'`, `'cut_resize'`, the last of which crops and resizes the volume back to its input shape instead of padding).
- On top of that shift, an intensity step is applied at the *same* `cut_off_position` on the *same* axis, as in `StepTransform` — this is what makes the combined transform resemble a real gated-acquisition seam, where the slab boundary shows both a brightness jump and a misregistration. Its offset is drawn internally and is not reported in `info`.

The output volume keeps the input's shape and container type (torch tensor or NumPy array), with or without a leading channel axis, though the step transforms return `float32` rather than the input dtype; when a `segmentation` is passed it is shifted by the same amount so image and labels stay aligned.

## Citation

If you use CTAug in your research, please cite:

```bibtex
@article{mohammadikazaj2026ctaug,
  title   = {A Unified Framework for Comprehensive Cardiac CT Segmentation and Phenotyping: Human-in-the-Loop Data Annotation, Vision Foundation Model Development, Multicenter Evaluation and Clinical Validation},
  author  = {Mohammadi Kazaj, Pooya and Weber, Leo Fridolin and Xie, Wen and Safavi-Naini, Seyed Amir Ahmad and Stark, Anselm and Baj, Giovanni and Mokhtari, Ali and Yoshida, Toshiya and Ryffel, Christoph and Okuno, Taishi and Akashi, Yoshihiro and Buechel, Ronny R. and Pilgrim, Thomas and Valenzuela, Waldo and Siontis, George C. M. and Xu, Xiaowei and Hundertmark, Moritz and Windecker, Stephan and Grani, Christoph and Shiri, Isaac},
  journal = {arXiv preprint arXiv:2607.11287},
  year    = {2026}
}
```

## Related repositories

CTAug is one component of a larger, openly released framework:

- [CCT-FM](https://github.com/AI-in-Cardiovascular-Medicine/CCT-FM) — model training, inference, and evaluation code
- [nnUZoo](https://github.com/ai-in-Cardiovascular-Medicine/nnUZoo) — segmentation model architectures
- [HolOrama](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama) — graphical user interface for cardiac CT

The associated annotated dataset is available on [Hugging Face](https://huggingface.co/datasets/AI-CVM/Cardiac-CT).

## License

See [LICENSE](https://github.com/AI-in-Cardiovascular-Medicine/CTAug/blob/main/LICENSE).
