#  Defect Image Generation using AI for better training

> **Generate synthetic defect images for low-sample datasets using advanced generative models — integrated with a full-stack web app (Next.js + FastAPI).**

---

##  Problem Statement

Manufacturers often have large datasets of **non-defective (good)** parts but very **few defect samples**. This imbalance leads to poor performance in machine learning models used for **automated defect detection**.

---

## 🎯 Objective

✅ **Generate realistic defect images** using generative models like **VAE, GAN, StyleGAN, and Diffusion Models**  
✅ **Improve classifier accuracy** by augmenting the defect class  
✅ Deploy a full-stack **web application** for end-to-end image upload → training → generation → download

---

## 🧠 Models Used

| Model      | Status       | Highlights |
|------------|--------------|------------|
| ✅ VAE        | ✔️ Tested     | Simple & fast, but blurry results |
| ✅ GAN        | ✔️ Tested     | Sharp images, but unstable training |
| ✅ StyleGAN   | ✔️ Tested     | High-quality outputs, data-hungry |
| ✅ DDPM       | ✔️ Final Model | Best results for small dataset |
| ✅ DefectGAN  | ✔️ Final Model | Custom GAN designed for defect patterns |

---

## 🖼️ Dataset

- **Custom Defect Dataset**
  - 24 RGB images
  - 700x700 resolution
  - Imbalanced: few defect images

---

##  Results

- Diffusion model produced **clear and realistic defects**
- Classifier trained on synthetic + real defects achieved **better accuracy** than using original data alone

| Metric |  With Synthetic Images |
|--------|------------------------|
| FID score | **86%** |

---

## Full Stack Web App

| Layer | Technology |
|-------|------------|
| **Frontend** | [Next.js](https://nextjs.org/) |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) |
| **ML Models** | PyTorch / TensorFlow |
| **Features** | Upload ZIP → Train Model → Generate Images → Download ZIP |

> Try the app: Upload a ZIP file of images → choose number of synthetic images → download results!

---



