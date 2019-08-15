# Towards Debiasing Fact Verification Models
- Symmetric evaluation set based on the FEVER (fact verification) dataset
- Regularization-based method

# Symmetric dataset 
To download the symmetric evaluation dataset from the EMNLP 2019 paper [Towards Debiasing Fact Verification Models](https://arxiv.org/abs/1908.05267) use this [link](https://github.com/TalSchuster/FeverSymmetric/raw/master/fever_symmetric_eval.jsonl).

As described in the paper, the cases are based on the [FEVER dataset](https://fever.ai/resources.html).

Each line in the jsonlines file contains:
* **id** - matches the FEVER id. For the new pairs, a suffix of *000000{2,3,4}* is added.
* **label** - SUPPORTS or REFUTES.
* **claim** - the claim.
* **evidence_sentence** - the evidence.

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
