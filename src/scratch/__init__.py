"""Multimodal product recommendation with image and text encoders trained FROM SCRATCH.

No pretrained weights anywhere: the image CNN, the text Transformer, the word embeddings and the
tokenizer vocabulary are all learned from the product catalog itself.

    data.py         dataset download, preprocessing, tokenizer, train/val/test split
    models.py       image CNN, text Transformer, fusion methods (incl. the proposed gated fusion)
    train.py        losses, training loop, evaluation of one run
    experiments.py  full study: optimizer tuning, optimizer comparison, method comparison, ablations
"""
