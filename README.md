# lameness-detection

--- 
**Code repository for the paper:**

Helena Russello, Rik van der Tol, Menno Holzhauer, Eldert J. van Henten, Gert Kootstra,
**Video-based automatic lameness detection of dairy cows using pose estimation and multiple locomotion traits,**
Computers and Electronics in Agriculture,
Volume 223,
2024,
109040,
ISSN 0168-1699,
https://doi.org/10.1016/j.compag.2024.109040.
(https://www.sciencedirect.com/science/article/pii/S0168169924004319)

---

### Usage

Install the conda environment:
```
conda env create -f environment-short.yml
```

Run the model:

```
python train_ml.py --config cfg/config.yml
```

You can add the path to your data in the `config.yml` file, and change the options as you wish.

The `data_files` contains examples of how to structure you dataset.

---

### Disclaimer

The code is provided as is, is not maintained, and no support is currently not provided.

If you have questions, send an email to 
helena [dot] russello [at] wur [dot] nl.

**!!! I may be slow to respond !!!**

---
If using (part of) the code, please cite:
```
@article{RUSSELLO2024109040,
title = {Video-based automatic lameness detection of dairy cows using pose estimation and multiple locomotion traits},
journal = {Computers and Electronics in Agriculture},
volume = {223},
pages = {109040},
year = {2024},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2024.109040},
url = {https://www.sciencedirect.com/science/article/pii/S0168169924004319},
author = {Helena Russello and Rik {van der Tol} and Menno Holzhauer and Eldert J. {van Henten} and Gert Kootstra}
}
```
