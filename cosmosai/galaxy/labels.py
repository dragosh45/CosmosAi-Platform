"""Shared galaxy label mapping used by training and inference code."""

# Stable class ID mapping for the current galaxy morphology labels.
LABEL_TO_ID = {
    "elliptical": 0,
    "spiral": 1,
    "lenticular": 2,
    "irregular": 3,
}

# Reverse lookup used when model predictions need human-readable labels.
ID_TO_LABEL = {
    label_id: label
    for label, label_id in LABEL_TO_ID.items()
}

# Allowed label set for manifest validation and readable error messages.
ALLOWED_LABELS = set(LABEL_TO_ID)

# Dataset split values currently allowed by the tiny local data contract.
ALLOWED_SPLITS = {
    "train",
    "val",
    "test",
}

