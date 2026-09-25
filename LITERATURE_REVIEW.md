# Literature review: multimodal product recommendation with image and text embeddings (2021–2025)

All 25 papers were checked against their publisher / arXiv / conference pages (links at the end).
"Results achieved" only states what each paper itself reports. "Limitation / gap" is **our analysis** with respect to
our project (end-to-end, from-scratch image + text encoders with gated fusion for product retrieval).

## A. Multimodal recommender models

| # | Title | Authors | Year / venue | Model used | Results achieved (as reported) | Limitation / gap (our analysis) |
|---|---|---|---|---|---|---|
| 1 | Mining Latent Structures for Multimedia Recommendation (LATTICE) | J. Zhang, Y. Zhu, Q. Liu, S. Wu, S. Wang, L. Wang | 2021, ACM MM | Modality-aware item–item graph learning + graph convolution on top of collaborative filtering | Outperforms state-of-the-art multimedia recommenders on three real-world datasets | Needs user–item interaction data; image/text features are pre-extracted by pretrained networks, not learned; costly graph learning (later criticised by FREEDOM) |
| 2 | Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation (MICRO) | J. Zhang et al. | 2022, IEEE TKDE | LATTICE + contrastive fusion of modality-shared and modality-specific information | Improves on LATTICE-style baselines on multiple datasets | Same dependence on interactions and frozen pretrained features; no single-modality query support |
| 3 | Multi-modal Graph Contrastive Learning for Micro-video Recommendation (MMGCL) | Z. Yi, X. Wang, I. Ounis, C. Macdonald | 2022, SIGIR | Graph contrastive learning with modality edge dropout and **modality masking**, modality-aware negative sampling | Better accuracy and faster convergence than SOTA on two micro-video datasets | Micro-video domain; user–item graphs; masking only used as augmentation, not for missing-modality queries |
| 4 | Bootstrap Latent Representations for Multi-modal Recommendation (BM3) | X. Zhou et al. | 2023, WWW | Self-supervised bootstrapping (dropout views), inter-/intra-modality alignment, no negatives | Beats prior models on three datasets (20K–200K nodes) with 2–9× less training time | Pre-extracted features; ID embeddings needed; not an item-to-item or text/photo search model |
| 5 | Multi-Modal Self-Supervised Learning for Recommendation (MMSSL) | W. Wei, C. Huang, L. Xia, C. Zhang | 2023, WWW | Adversarial modality-aware structure learning + cross-modal contrastive learning | Better than SOTA multimodal recommenders | Complex (adversarial) training; interaction-based; features not learned from raw pixels / words |
| 6 | A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation (FREEDOM) | X. Zhou, Z. Shen | 2023, ACM MM | Frozen item–item graph + denoised user–item graph | Shows LATTICE's graph learning is unnecessary; competitive accuracy with much lower cost | Relies on fixed pretrained features; fusion is at graph level, not per-feature |
| 7 | Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation (DRAGON) | H. Zhou, X. Zhou, Z. Shen | 2023, ECAI | User–user + item–item homogeneous graphs, attentive concatenation fusion | Reports that common fusion choices can hurt; improves over graph baselines | Fusion by concatenation; interaction-dependent; no content-only (cold-start) queries |
| 8 | LGMRec: Local and Global Graph Learning for Multimodal Recommendation | Z. Guo et al. | 2024, AAAI | Local graph + global hypergraph learning, group-wise contrastive alignment | Improves over graph-based multimodal recommenders | Heavy graph machinery; pretrained features; not designed for missing modalities |
| 9 | MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation | J. Xu, Z. Chen, S. Yang, J. Li, H. Wang, E. C. H. Ngai | 2025, AAAI | Multi-level cross-modal alignment guided by ID embeddings + feature-enhancement task | Reduces the modality gap while keeping interaction information | ID-guided alignment needs interaction history; encoders not trained end-to-end |
| 10 | Modality-Balanced Learning for Multimedia Recommendation (CKD) | J. Zhang, G. Liu, Q. Liu, S. Wu, L. Wang | 2024, ACM MM | Counterfactual knowledge distillation to balance modalities | Shows **text-only can outperform multimodal** ("1+1<2"); CKD consistently improves late- and early-fusion backbones | Distillation adds training cost; still built on pre-extracted features. **Directly explains our text-dominance finding** |

## B. Critical analyses and surveys

| # | Title | Authors | Year / venue | Model used | Results achieved (as reported) | Limitation / gap (our analysis) |
|---|---|---|---|---|---|---|
| 11 | Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited | Yuan et al. | 2023, SIGIR | Compares ID-based (IDRec) vs modality-based (MoRec) recommenders with text/image encoders trained end-to-end | Modality-based models can rival ID-based ones when encoders are trained end-to-end | Uses large pretrained encoders; does not study fusion design or missing modalities |
| 12 | Do We Really Need to Drop Items with Missing Modalities in Multimodal Recommendation? | D. Malitesta, E. Rossi, C. Pomo, T. Di Noia, F. D. Malliaros | 2024, CIKM (short) | Graph-based imputation of missing modality features (pre-processing) | Dropping items is unnecessary and harmful; imputation works with any recommender | Imputation happens before training; does not handle a missing modality in the *query* |
| 13 | Does Multimodality Improve Recommender Systems as Expected? A Critical Analysis and Future Directions | Zhou et al. | 2025, arXiv | Large re-evaluation of multimodal recommenders | Gains are modest and context-dependent; **text matters more in e-commerce**; ensembles beat fusion; only 29.3% of 41 papers fully reproducible | Analysis paper, no new model. Supports our emphasis on reproducibility and honest evaluation |
| 14 | Multimodal Recommender Systems: A Survey | Q. Liu, J. Hu, Y. Xiao, X. Zhao, J. Gao, W. Wang, Q. Li, J. Tang | 2024, ACM Computing Surveys | Survey (feature interaction, feature enhancement, model optimisation) | Taxonomy of multimodal recommenders | Survey; little on from-scratch encoders or query-time missing modalities |
| 15 | Multimodal Pretraining, Adaptation, and Generation for Recommendation: A Survey | Q. Liu, J. Zhu, Y. Yang, Q. Dai, Z. Du, X.-M. Wu, Z. Zhao, R. Zhang, Z. Dong | 2024, KDD | Survey of pretraining, adaptation and generation for recommendation | Maps the trend towards large pretrained multimodal models | Focus on large pretrained models; resource-light from-scratch training is under-explored |

## C. Vision–language models and fashion retrieval

| # | Title | Authors | Year / venue | Model used | Results achieved (as reported) | Limitation / gap (our analysis) |
|---|---|---|---|---|---|---|
| 16 | Learning Transferable Visual Models From Natural Language Supervision (CLIP) | A. Radford et al. | 2021, ICML | Dual encoder (image + text) trained contrastively on 400M image–text pairs | Zero-shot transfer across 30+ vision datasets, often competitive with supervised models | Needs web-scale data and compute; no fusion into one item embedding; general, not product-specific |
| 17 | Sigmoid Loss for Language Image Pre-Training (SigLIP) | X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer | 2023, ICCV | Image–text pre-training with a pairwise sigmoid loss | 84.5% ImageNet zero-shot (SigLiT) with 4 TPUv4 chips in 2 days; better at small batch sizes | Still large pretraining; in our reference test its text tower did poorly on short product titles |
| 18 | Contrastive Language and Vision Learning of General Fashion Concepts (FashionCLIP) | P. J. Chia et al. | 2022, Scientific Reports | CLIP fine-tuned on fashion image–caption pairs | Strong transferable fashion representations across many tasks and datasets | Depends on pretrained CLIP + large proprietary catalog; no explicit fusion or missing-modality handling |
| 19 | FashionViL: Fashion-Focused Vision-and-Language Representation Learning | X. Han et al. | 2022, ECCV | Modality-agnostic Transformer with multi-view contrastive and pseudo-attribute pre-training | State of the art on five fashion downstream tasks | Heavy pre-training; large model; not aimed at lightweight recommendation |
| 20 | Effective Conditioned and Composed Image Retrieval Combining CLIP-based Features (CLIP4Cir) | A. Baldrati, M. Bertini, T. Uricchio, A. Del Bimbo | 2022, CVPR (demo) | Combiner network over frozen CLIP image + text features | Composed "image + text change" retrieval (Best Demo Honorable Mention) | Built on pretrained CLIP. Relevant to our photo + text query, where we found text can override the photo |

## D. Missing modalities and modality imbalance

| # | Title | Authors | Year / venue | Model used | Results achieved (as reported) | Limitation / gap (our analysis) |
|---|---|---|---|---|---|---|
| 21 | Are Multimodal Transformers Robust to Missing Modality? | M. Ma, J. Ren, L. Zhao, D. Testuggine, X. Peng | 2022, CVPR | Study of multimodal Transformers + learned fusion-strategy search | Transformers are sensitive to missing modalities; the best fusion strategy is dataset-dependent | Classification tasks, not retrieval / recommendation |
| 22 | Multimodal Prompting with Missing Modalities for Visual Recognition | Y.-L. Lee, Y.-H. Tsai, W.-C. Chiu, C.-Y. Lee | 2023, CVPR | Missing-aware prompts on a pretrained multimodal Transformer (ViLT) | Handles missing modalities while training under 1% of the parameters | Needs a large pretrained backbone; recognition task |
| 23 | Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling (ShaSpec) | H. Wang, Y. Chen, C. Ma, J. Avery, L. Hull, G. Carneiro | 2023, CVPR | Shared + modality-specific features, distribution alignment, residual fusion | Beats SOTA by large margins (e.g. +3–5% on BraTS2018 tumour segmentation) | Medical / classification domain; not a recommendation or retrieval setting |
| 24 | Balanced Multimodal Learning via On-the-fly Gradient Modulation (OGM-GE) | X. Peng, Y. Wei, A. Deng, D. Wang, D. Hu | 2022, CVPR (oral) | Gradient modulation per modality + generalisation enhancement noise | Considerable gains over common fusion methods on audio-visual tasks | Classification with a dominant modality; not tested for retrieval / recommendation |
| 25 | On Uni-Modal Feature Learning in Supervised Multi-Modal Learning | Du et al. | 2023, ICML | Uni-Modal Ensemble / Uni-Modal Teacher; identifies "modality laziness" | Simple late-fusion variants match complex fusion methods on several datasets | Classification; explains why a weaker modality (our image branch) is under-trained |

## What the literature leaves open: where our project is novel

1. **Recommenders use frozen, pretrained features.** Papers 1–10 feed pre-extracted image and text features (from pretrained CNNs and sentence encoders) into user–item graphs. We instead learn **both encoders end-to-end from raw pixels and words**, with our own tokenizer and **no pretrained weights**. That is a resource-light setting (3–4M parameters, trained on one free GPU) which the surveys (14, 15) note is under-explored.
2. **Content-only retrieval versus interaction graphs.** Most models need user–item interaction histories. Ours works from content alone (cold start), supporting four query types in one model: similar item, text, photo, and photo + text.
3. **Missing modality at query time.** Missing-modality work (12, 21–23) handles classification, or imputes features before training. We handle a missing modality **in the query** with learned null vectors and modality dropout, and measure it (photo-only and text-only search).
4. **Per-feature gated fusion inside the item embedding.** Graph papers fuse at graph or score level (6, 7). We learn a per-dimension, per-item gate between image and text. Note: the gating idea goes back to Gated Multimodal Units (Arevalo et al., 2017, pre-2021), so our claim is its **use and analysis for from-scratch product retrieval**, not the gate itself.
5. **Rigorous, reproducible evaluation.** Tuned optimizers (SGD / Adam / AdamW), 3 seeds, paired bootstrap significance, ablations, released code: this directly answers the reproducibility problem raised in paper 13.

## Our finding that connects to the literature, and the strongest route to more novelty

Our text-only model matched or beat fusion on some tasks. This is the **modality-imbalance** effect reported in papers 10, 24 and 25, and paper 13 found text dominates in e-commerce. Two concrete extensions would make the paper clearly novel:
- **(a) A modality-balanced gated fusion.** Add gradient modulation (as in 24) or uni-modal teachers (as in 25) to the gated model trained from scratch, and test whether the image branch then helps on type + colour. No recommendation paper above does this with from-scratch encoders.
- **(b) A noisy- / missing-title benchmark.** Evaluate with titles partially masked or corrupted. Paper 12 motivates missing modalities, but nobody measures from-scratch fusion under title corruption. This is where the gate should shine over text-only.

## Sources
1. https://arxiv.org/abs/2104.09036 · https://dl.acm.org/doi/10.1145/3474085.3475259
2. https://arxiv.org/abs/2111.00678 · https://ieeexplore.ieee.org/document/9950351/
3. https://dl.acm.org/doi/10.1145/3477495.3532027
4. https://arxiv.org/abs/2207.05969 · https://dl.acm.org/doi/10.1145/3543507.3583251
5. https://arxiv.org/pdf/2302.10632 · https://dl.acm.org/doi/10.1145/3543507.3583206
6. https://arxiv.org/pdf/2211.06924 · https://doi.org/10.1145/3581783.3611943
7. https://arxiv.org/abs/2301.12097 · https://ebooks.iospress.nl/doi/10.3233/FAIA230631
8. https://ojs.aaai.org/index.php/AAAI/article/view/28688
9. https://ojs.aaai.org/index.php/AAAI/article/view/33408
10. https://arxiv.org/pdf/2408.06360 · https://dl.acm.org/doi/10.1145/3664647.3680626
11. https://dl.acm.org/doi/abs/10.1145/3539618.3591932
12. https://arxiv.org/abs/2408.11767 · https://dl.acm.org/doi/10.1145/3627673.3679898
13. https://arxiv.org/pdf/2508.05377
14. https://dl.acm.org/doi/abs/10.1145/3695461 · https://arxiv.org/pdf/2302.03883
15. https://arxiv.org/abs/2404.00621
16. https://proceedings.mlr.press/v139/radford21a/radford21a.pdf
17. https://arxiv.org/abs/2303.15343 · https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html
18. https://www.nature.com/articles/s41598-022-23052-9 · https://arxiv.org/abs/2204.03972
19. https://arxiv.org/abs/2207.08150 · https://link.springer.com/chapter/10.1007/978-3-031-19833-5_37
20. https://www.semanticscholar.org/paper/Effective-conditioned-and-composed-image-retrieval-Baldrati-Bertini/744c42180b9721c1c15b753ba9c00b019b5dfec2
21. https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html
22. https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Multimodal_Prompting_With_Missing_Modalities_for_Visual_Recognition_CVPR_2023_paper.html
23. https://arxiv.org/abs/2307.14126 · https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.html
24. https://arxiv.org/abs/2203.15332 · https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html
25. https://arxiv.org/abs/2305.01233 · https://proceedings.mlr.press/v202/du23e/du23e.pdf
