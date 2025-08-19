# R2GEN-Finetuning

This repository contains the **fine-tuning implementation of R2Gen** for radiology report generation.  
It is based on the original [R2Gen model](https://github.com/cuhksz-nlp/R2Gen), which introduced the **Memory-driven Transformer** for generating radiology reports (EMNLP 2020).  

This repository was developed by **Ozan Orhan** and **İrfan Şenel**.  

Our contribution focuses on **fine-tuning** the model with biomedical datasets (e.g., IU-XRAY) and optimizing decoding parameters to improve clinical accuracy and linguistic quality.  

## Project Overview
- **R2Gen** generates radiology report drafts from chest X-ray images.  
- Fine-tuning improves the quality of generated reports.  
- Performance is evaluated with **BLEU, ROUGE, METEOR, and CIDEr** metrics.  

## Datasets
- **IU-XRAY** (Indiana University Chest X-Ray Collection)  
- **MIMIC-CXR** (optional, supported as in original implementation)  

Datasets should be prepared as described in the [original R2Gen repository](https://github.com/cuhksz-nlp/R2Gen).  

## References
If you use this work, please also cite the original R2Gen paper:  

```bibtex
@inproceedings{chen-emnlp-2020-r2gen,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
