# Endo-NeRF++

### Uncertainty-Aware Sampling and Multi-Resolution Encoding for Dynamic Surgical Scene Reconstruction

<p align="center">
  <img src="assassets/Diagram final.jpg" width="800"/>
</p>

---

## 📌 Overview
Endo-NeRF++ is a scalable framework for dynamic surgical scene reconstruction.  
It extends EndoNeRF by integrating:

- Multi-resolution hash encoding for compact scene representation  
- Feature blending across temporal dimensions  
- Uncertainty-aware modeling for robust reconstruction  

These improvements enable efficient learning and better performance in deformable and texture-sparse surgical environments.

---

## 🧠 Method
Our approach combines:
- Temporal deformation field (Direct Temporal NeRF)
- Multi-resolution hash encoding
- Temporal feature blending
- Uncertainty-aware learning

---

## 📊 Results
Endo-NeRF++ improves reconstruction quality and robustness compared to baseline EndoNeRF.

(Add your results images here later)

---

## 🔗 Project Page
https://github.com/gousiya26-I/EndoNerf-plusplus

---

## ⚙️ Installation
```bash
git clone https://github.com/gousiya26-I/EndoNerf-plusplus.git
cd EndoNerf-plusplus
pip install -r requirements.txt

## 🚀 Running Different Variants

### 1. Baseline EndoNeRF

For the original EndoNeRF implementation, use:

- `run_endonerf.py`
- `run_endonerf_helpers.py`
- `Configs/example.txt`

Example:

```bash
python run_endonerf.py --config Configs/example.txt
```

---

### 2. Multi-Resolution Encoding

For the multi-resolution hash encoding version, use:

- `Multiresolution_Encoding_run_endonerf.py`
- `Multiresolution_Encoding_run_endonerf_helpers.py`
- Corresponding configuration file

Example:

```bash
python Multiresolution_Encoding_run_endonerf.py --config Configs/multiresolution_example.txt
```

---

### 3. Uncertainty-Aware Estimation

For uncertainty-aware reconstruction, use:

- `Uncertainty_Estimation_run_endonerf.py`
- `Uncertainty_Estimation_run_endonerf_helpers.py`
- Corresponding configuration file

Example:

```bash
python Uncertainty_Estimation_run_endonerf.py --config Configs/uncertainty_example.txt
```

---

### 4. Adaptive Sampling

For adaptive sampling, use:

- `Adaptive_Sampling_run_endonerf.py`
- `Adaptive_Sampling_run_endonerf_helpers.py`
- Corresponding configuration file

Example:

```bash
python Adaptive_Sampling_run_endonerf.py --config Configs/adaptive_sampling_example.txt
```

---

### Repository Structure

```text
Configs/
├── example.txt
├── multiresolution_example.txt
├── uncertainty_example.txt
└── adaptive_sampling_example.txt

run_endonerf.py
run_endonerf_helpers.py

Multiresolution_Encoding_run_endonerf.py
Multiresolution_Encoding_run_endonerf_helpers.py

Uncertainty_Estimation_run_endonerf.py
Uncertainty_Estimation_run_endonerf_helpers.py

Adaptive_Sampling_run_endonerf.py
Adaptive_Sampling_run_endonerf_helpers.py
```
