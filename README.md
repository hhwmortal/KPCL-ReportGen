# KPCL-ReportGen
This repository provides the official codes of KPCL and the PED-Xray dataset.
## Overview of KPCL
In this work, a novel framework for pediatric radiology report generation that combines a Parallel Attention Module with Clinical Key Sentence Guidance (PAM-KG) and a text-guided Multi-Granularity Contrastive Learning (MGCL) strategy is proposed. The PAM-KG module extracts global and local semantic cues from images, enabling the model to focus on regions associated with potential abnormalities.
<img width="1647" height="1064" alt="image" src="https://github.com/user-attachments/assets/b8ad704f-98fa-4861-a86c-a773d20ebc09" />


## Dataset Details
The Pediatric Chest X-ray (PED-Xray) dataset consists of 5850 high-quality images and 2925 reports from 2,925 patients aged 1-12 years, and the X-ray images are saved in JPG format. The original data includes two views of each patient (i.e., frontal and lateral views), and a Chinese radiology report written by a clinician with more than 5 years of experience. The dataset is divided into a training set, a validation set, and a test set with a ratio of 7:1:2. Notably, it is ensured that there is no overlapping of data between these sets, and basic preprocessing and manual inspection of the data are performed. The manual inspection includes deleting reports with obvious errors and anonymizing patient information to ensure the reliability of the training results and medical privacy. In addition, all acquired images and corresponding textual reports are examined and verified by a senior radiologist with more than ten years of clinical experience. The data collected from Xiangyang Central Hospital strictly adhered to the ethical principles outlined in the Declaration of Helsinki, and approval is obtained from the Medical Ethics Committee of Xiangyang Central Hospital (No. 2025-186). Written informed consent is waived due to the retrospective nature of the study, as permitted by the committees. The dataset can also be downloaded using the following links:
Google Drive:[link](https://drive.google.com/file/d/1luCzRfpaE9du1eS7513-pY5pqcAeAcKv/view?usp=sharing)
<img width="2138" height="1054" alt="c8b48a8f-4093-4280-8585-590b3893cb6b" src="https://github.com/user-attachments/assets/122c97dc-8ab8-4d13-bcd5-cc2117e7c255" />
## Acknowledgment
This research was funded by the Guangdong Basic and Applied Basic Research Foundation (No. 2021B1515130003), the Key Research and Development Plan of Hubei Province (No. 2022BCE034), the Natural Science Foundation of Hubei Province (No. 2025AFD118) and the Scientific Research Program of Hubei Provincial Department of Education (No. B2023150).
Our project references the codes in the following repos. Thanks for their works and sharing.
[R2Gen](https://github.com/zhjohnchan/R2Gen)
