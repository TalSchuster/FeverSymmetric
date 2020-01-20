# Towards Debiasing Fact Verification Models
- Symmetric evaluation set based on the FEVER (fact verification) dataset
- Regularization-based method

# Symmetric dataset 
To download the symmetric evaluation dataset from the EMNLP 2019 paper [Towards Debiasing Fact Verification Models](https://arxiv.org/abs/1908.05267) use this [link](https://raw.githubusercontent.com/TalSchuster/FeverSymmetric/master/symmetric_v0.1/fever_symmetric_generated.jsonl).

## Version 0.2
We release a version that includes new cases. This version is split to dev (708 pairs) and test (712 pairs) to allow models to use the dev set for hyperparameter tuning. 

## Version 0.1
The version used in "Towards Debiasing Fact Verification Models" paper.

We've implemented the baseline and the reweighted version on the latest version of the pytorch-transformers repository ([link](https://github.com/TalSchuster/pytorch-transformers)). Since the test set is small, there are some random variations across different runs using different servers/GPUs. Therefore, to allow better comparison across methods, we've run the training five times with different random seeds and report the average and std of the runs:

|             | Symmetric (generated) | Fever DEV      | delta |
|-------------|-----------------------|----------------|-------|
| baseline    | 57.46 (+/-1.6)        | 85.85 (+/-0.5) |       |
| re-weighted | 61.62 (+/-1.2)        | 85.95 (+/-0.5) | 4.16  |

## Dataset format
As described in the paper, the cases are based on the [FEVER dataset](http://fever.ai/resources.html).

Each line in the jsonlines file contains:
* **id** - matches the FEVER id. For the new pairs, a suffix of *000000{2,3,4}* is added.
* **label** - SUPPORTS or REFUTES.
* **claim** - the claim.
* **evidence_sentence** - the evidence.

# Training
Our processed FEVER training data is available [here](https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl). It includes only cases that can be validated with a single evidence sentence. The evidence sentences for the NOT ENOUGH INFORMATION sampled from the NSMN retrieval model.

The processed FEVER evaluation data is available [here](https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.json).

In order to train the baseline model, use the run `bash train_baseline.sh`.

To use the re-weighted training, add the `weighted_loss` flag.

# Citation

If you find this repo useful, please cite our paper.

```
@InProceedings{schuster2019towards,
  author = 	"Schuster, Tal and
  			Shah, Darsh J and
  			Yeo, Yun Jie Serene and
  			Filizzola, Daniel and
  			Santus, Enrico and
  			Barzilay, Regina", 			
  title = 	"Towards Debiasing Fact Verification Models",
  booktitle = 	"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
  year = 	"2019",
  publisher = 	"Association for Computational Linguistics",
  url = 	"https://arxiv.org/abs/1908.05267"
}
```

## Related papers
* [Automatic Fact-guided Sentence Modification](https://arxiv.org/abs/1909.13838)
* [Simple but effective techniques to reduce biases](https://arxiv.org/abs/1909.06321)
* [FEVER: a large-scale dataset for Fact Extraction and VERification](https://arxiv.org/abs/1803.05355)
* [Adversarial attacks against Fact Extraction and VERification](https://arxiv.org/abs/1903.05543)
