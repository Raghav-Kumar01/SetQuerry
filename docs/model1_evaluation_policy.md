# Model 1 Evaluation Policy

Model 1 uses the fixed answer vocabulary constructed from training answers
only (6,359 classes).

During validation and test evaluation, samples whose ground-truth answer is
absent from that training vocabulary are excluded from answer-accuracy
calculation. They are counted and reported as `excluded_unseen_answers`.

Known-answer samples are evaluated normally. The raw manifests and images are
not modified, and validation/test answers are never added to the classifier
vocabulary. No `<unk>` answer class or answer remapping is used.

Training remains strict: a training answer absent from the training
vocabulary raises `ValueError`.
