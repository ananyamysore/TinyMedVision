# Dataset provenance and access

## Recommended documented benchmark: PneumoniaMNIST

TinyMedVision deliberately does not download data. If you choose to evaluate on
PneumoniaMNIST, obtain it yourself from the [official MedMNIST Zenodo record](https://doi.org/10.5281/zenodo.10519652)
or use the official [MedMNIST dataset guide](https://medmnist.com/). The official
MedMNIST metadata list PneumoniaMNIST under CC BY 4.0 and describe a binary task
(0 normal, 1 pneumonia) based on 5,856 pediatric chest radiographs. Its
PneumoniaMNIST representation is center-cropped/resized to 28×28 grayscale.
See the [official metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
and [project licensing notes](https://github.com/MedMNIST/MedMNIST/blob/main/README.md).
Review those upstream pages and the applicable CC BY 4.0 terms before use,
especially redistribution and attribution requirements. TinyMedVision does not
redistribute image data.

The predefined MedMNIST train/validation/test split is inherited from the source
dataset workflow. The package does not claim that the split is patient-disjoint;
the public metadata cited here do not establish patient-level separation. Thus
patient-level leakage cannot be ruled out from the data alone. Preserve any
patient/study identifiers you are authorized to use and supply them as
`group_id` in the manifest. `tinymedvision.splitting` rejects group overlap for
declared splits and offers grouped split generation.

The 28×28 preprocessing removes substantial image detail and is suited to
lightweight method development, not diagnostic interpretation. Labels can be
noisy, the classes are imbalanced, the cohort is pediatric, and source-specific
acquisition or workflow cues can drive apparent model performance. There is no
demographic metadata here to support subgroup fairness analysis. Neither a
successful software smoke test nor internal benchmark scores imply clinical
utility or external validity.

## Local manifest format

Use a UTF-8 CSV with `path,label` and optionally `group_id,split`. Relative image
paths are resolved relative to the manifest file. Labels are integer class IDs;
the binary evaluation API expects 0=normal and 1=pneumonia. Group IDs should be
stable patient, study, or source identifiers where permitted. Never put patient
identifiers in a public repository; keep manifests and images local.

The initial loader accepts standard Pillow-readable image formats and converts
to grayscale. Dataset-specific archives need to be extracted or adapted to
this manifest format by the data owner.
