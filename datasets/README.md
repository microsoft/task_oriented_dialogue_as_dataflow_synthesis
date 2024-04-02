# Datasets

This directory contains the conversational semantic parsing datasets we used for the experiments of the following papers:

```bibtex
@article{SMDataflow2020,
  author = {{Semantic Machines} and Andreas, Jacob and Bufe, John and Burkett, David and Chen, Charles and Clausman, Josh and Crawford, Jean and Crim, Kate and DeLoach, Jordan and Dorner, Leah and Eisner, Jason and Fang, Hao and Guo, Alan and Hall, David and Hayes, Kristin and Hill, Kellie and Ho, Diana and Iwaszuk, Wendy and Jha, Smriti and Klein, Dan and Krishnamurthy, Jayant and Lanman, Theo and Liang, Percy and Lin, Christopher H. and Lintsbakh, Ilya and McGovern, Andy and Nisnevich, Aleksandr and Pauls, Adam and Petters, Dmitrij and Read, Brent and Roth, Dan and Roy, Subhro and Rusak, Jesse and Short, Beth and Slomin, Div and Snyder, Ben and Striplin, Stephon and Su, Yu and Tellman, Zachary and Thomson, Sam and Vorobev, Andrei and Witoszko, Izabela and Wolfe, Jason and Wray, Abby and Zhang, Yuchen and Zotov, Alexander},
  title = {Task-Oriented Dialogue as Dataflow Synthesis},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {8},
  pages = {556--571},
  year = {2020},
  month = sep,
  url = {https://doi.org/10.1162/tacl_a_00333},
}

@inproceedings{SMValueAgnosticParsing2021,
  author = {Platanios, Emmanouil Antonios and Pauls, Adam and Roy, Subhro and Zhang, Yuchen and Kyte, Alex and Guo, Alan and Thomson, Sam and Krishnamurthy, Jayant and Wolfe, Jason and Andreas, Jacob and Klein, Dan},
  title = {Value-Agnostic Conversational Semantic Parsing},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  month = aug,
  year = {2021},
  address = {Online},
  publisher = {Association for Computational Linguistics},
}
```

There are three datasets:

- **SMCalFlow-1.0:** This is the dataset released with the [**Task-Oriented Dialogue as Dataflow Synthesis** (TACL 2020)](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00333) paper.
- **SMCalFlow-2.0:** This is an updated version of the dataset released with the [**Task-Oriented Dialogue as Dataflow Synthesis** (TACL 2020)](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00333) paper, which removed a very small number of incorrectly annotated examples, dropped argument names for positional arguments (so that the programs are shorter), and added inferred type arguments for type-parameterized functions that were missing in the original SMCalFlow data.
- **TreeDST:** This is a modified version of the [TreeDST dataset]([apple/ml-tree-dst (github.com)](https://github.com/apple/ml-tree-dst)) which has been converted to the Lispress representation used for SMCalFlow 2.0, and transformed to make use of the `refer` and `revise` meta-computation operators. The transformation is described in the appendix of the paper referenced above.

Furthermore, compared to the original release of the SMCalFlow dataset, these two datasets also provide programs which have been fully annotated with argument names for all function arguments and types for all expressions after running a Hindley-Milner based type inference algorithm (also described in the aforementioned paper). These programs are included in the new `fully_typed_lispress` field in the JSON objects that correspond to dialogue turns. It is not recommended to use these programs directly with simple Seq2Seq baselines because they are very verbose and the information they additional information they contain can be derived directly from the `lispress` programs by running type inference. That is also why the `lispress` programs are the ones used by the official evaluation script in SMCalFlow leaderboard.

Note that the version uploaded before June 28, 2021 contained some minor errors. You should
re-download the datasets if you downloaded the datasets before that date.
