# LLM Role-Play Bias Experiment

This repository contains the code, data, and analysis for the term paper:

**“Cultural Stereotypes in Role-Playing LLMs: A Light Experimental Look into Bias and Representation”**  
University of Trier · M.Sc. Natural Language Processing · Summer Semester 2025   
Advisor: Prof. Dr. Achim Rettinger   

---

## Overview

Large language models (LLMs) are increasingly used in **role-playing contexts** (e.g., teacher, parent, service worker). While useful, these interactions may reproduce **cultural stereotypes**.  
This project explores:

- How stereotypes surface when an LLM role-plays across **nationalities** and **roles**.  
- Whether a short **mitigation instruction** reduces these stereotypes.  
- How simple metrics (Stereotype Index + style features) make patterns visible.  

---

## Method

- **Model**: OpenAI GPT-4o (2024-05-13 API release)  
- **Nationalities**: German, Japanese, Brazilian, Turkish  
- **Roles**: teacher, parent, service worker/friend  
- **Scenarios**:  
  1. Career advice to a teenager  
  2. Handling a café complaint  
  3. Comforting a stressed friend  
- **Conditions**:  
  - *Plain*: direct role-play  
  - *Mitigated*: same prompt with “avoid stereotypes, be respectful”  
- **Replicates**: 5 per cell → **360 outputs total**  

---

## Metrics

- **Stereotype Index (SI)**: stereotype markers per 100 tokens  
- **Style features**: hedges, deontic modality, affect words, directives, identity mentions  
- **Mitigation effect (ΔSI)**: plain vs. mitigated difference  
- **Audit reliability**: 25% manual audit, double-coded subset (Cohen’s κ = 0.75)  

---

## Results (Short Preview)

- **Cross-cultural variation**: Turkish and German responses carried the strongest stereotypes; Japanese the lowest.  
- **Motifs**: Frequent identity phrases (“as a German…”, “as a Brazilian…”) and cultural clichés (strictness, politeness, emotionality).  
- **Role effects**: Parental roles amplified stereotypes most; teachers lowest except in German contexts.  
- **Mitigation**: A single prompt line reduced SI in nearly all cases (median ΔSI overall = 0.31).  
- **Reliability**: Manual coding confirmed substantial agreement (88%, κ = 0.75).  

---

## Ethics and Limitations

- Scenarios were everyday, low-risk, and culturally neutral.  
- Dataset was small (4 nationalities × 3 roles × 3 scenarios).  
- Only English outputs, single model (GPT-4o).  
- Mitigation reduced but did not fully eliminate stereotypes.  

**Future work:** Extend to multilingual prompts, larger role sets, multiple models, and more advanced mitigation methods.  

---

## Repository Contents

- `llm_roleplay_experiment.ipynb` – main experiment codes  
- `llm_roleplay_experiment.csv` – raw outputs  
- `llm_roleplay_experiment_annotated.csv` – annotated data (SI + features)  
- `llm_roleplay_summary.csv` – summary stats  
- Additional CSVs: motif counts, medians/IQR, audit files  


⚠️ **Note:** Due to university policy, the full paper **cannot be shared publicly**. This repository only provides code, datasets, and reproducibility materials.  

---

## Author

- Abdullah Orzan — s2aborza@uni-trier.de
