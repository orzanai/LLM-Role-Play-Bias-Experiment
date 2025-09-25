# LLM Role-Play Bias Experiment

This repository contains the code, data, and analysis for the term paper:

**“Cultural Stereotypes in Role-Playing LLMs: A Light Experimental Look into Bias and Representation”**  
University of Trier · M.Sc. Natural Language Processing · Summer Semester 2025

---

## Overview

Large language models (LLMs) are often used in role-playing scenarios, such as acting as a teacher, a parent, or a service worker. While useful, these role-plays may reproduce cultural stereotypes.  
This project explores:

- How stereotypes appear when an LLM is asked to role-play across nationalities and roles.  
- Whether adding a short mitigation instruction reduces these stereotypes.  
- How simple metrics (Stereotype Index and style features) make these patterns visible.  

---

## Method

- **Model**: OpenAI GPT-4o (2024-05-13 API release)  
- **Nationalities**: German, Japanese, Brazilian, Turkish  
- **Roles**: teacher, parent, service worker/friend  
- **Scenarios**: career advice, handling a café complaint, comforting a stressed friend  
- **Conditions**:  
  - *Plain*: direct role-play prompt  
  - *Mitigated*: same prompt with “avoid stereotypes, be respectful” instruction  
- **Replicates**: 5 per cell → total 360 outputs  

---

## Metrics

The analysis tracks:

- **Stereotype Index (SI)**: stereotype markers per 100 tokens  
- **Style features**: hedges, deontic modality, affect words, directives, identity mentions  
- **Mitigation effect**: ΔSI between plain and mitigated runs  

---

## Results (Short Preview)

- Stereotypes were strongest in German and Brazilian parental roles.  
- Japanese and Turkish roles showed fewer stereotypes but leaned on politeness or emotional tone.  
- Adding one mitigation line reduced SI sharply, sometimes to zero.  

---

## Author

- Abdullah Orzan — s2aborza@uni-trier.de
