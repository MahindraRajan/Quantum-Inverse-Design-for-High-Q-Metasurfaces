# Hybrid Quantum Inverse Design for High-Q Metasurfaces

**LaSt-QGAN** (Latent-Style Quantum GAN) is a **hybrid quantum-classical framework** that performs inverse design of metasurfaces to achieve **tailored high-Q (up to 10⁴)** narrow-band absorption and controlled unidirectionality, with major reductions in data and runtime.

---

## 🌟 Highlights
- **High-Q Fano resonances:** Q ≈ 10⁴ from models trained around Q ≈ 10³  
- **Data-efficient:** ~40× fewer samples than classical GAN baselines  
- **Fast training:** ~10× runtime reduction vs. classical GANs  
- **Manufacturable designs:** Real-material lookup maps predicted indices to nearest dielectrics with minimal re-sim error  
- **Targets:** Narrow-band absorption, tailored lineshapes, optional unidirectionality

---

## 🧠 Method (LaSt-QGAN)
- **Encoder:** Pretrained VAE (β-VAE / IWAE) compresses metasurface images (64×64×3) into latent features.  
- **Quantum Generator:** Style-conditioned variational quantum circuits (PennyLane) take latent noise + conditional vector (Fano/Lorentzian params) to synthesize features.  
- **Discriminator:** Classical MLP guides training to match target spectra/features.  
- **RGB Encoding:**  
  - **R:** Metal (Drude) plasma frequency / geometry (MIM)  
  - **G:** Dielectric refractive index / geometry (hybrid)  
  - **B:** Dielectric thickness (both)  
- **Geometries:** MIM and hybrid dielectric unit cells; COMSOL used for forward validation.

---

## 📊 Results (Summary)
- **Spectral fidelity:** Lower MSE than classical GANs (≈10⁻⁴ vs. 10⁻³)  
- **Generalization:** Generates **higher-Q** designs than seen in training  
- **Material substitution:** Nearest real materials preserve response with low error

> Example targets: Lorentzian and Fano absorption lines (4–12 μm), angle control optional; designs validated via FEM (COMSOL).

---

## 🔬 Citation
If this work helps your research, please cite the corresponding paper (preprint/manuscript):

Warrier, S. R., & Dontabhaktuni, J. (2025). Hybrid Quantum-Classical Inverse Design of Metasurfaces for Tailored Narrow Band Absorption. arXiv preprint ([arXiv:2507.18127][https://arxiv.org/abs/2507.18127])

```bibtex
@article{warrier2025hybrid,
  title={Hybrid Quantum-Classical Inverse Design of Metasurfaces for Tailored Narrow Band Absorption},
  author={Warrier, Sreeraj Rajan and Dontabhaktuni, J.},
  journal={arXiv preprint arXiv:2507.18127},
  year={2025},
  url={https://arxiv.org/abs/2507.18127}
}
```





