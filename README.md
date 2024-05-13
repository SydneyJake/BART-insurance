# BART-insurance

1. The BART and XGB F1 scores can be obtained by running ```Rscript bart_f1.r``` and ```Rscript xgb_f1.r```  from the terminal.
2. Once both results have been calculated the plots are generated using ```Rscript plot_f1.r```.


# Data

The file ```data.csv``` contain a sample of 50000 rows from the dataset: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Base.csv. Data has been sampled as to preserve the class imbalance.

```
@article{jesusTurningTablesBiased2022,
  title={Turning the {{Tables}}: {{Biased}}, {{Imbalanced}}, {{Dynamic Tabular Datasets}} for {{ML Evaluation}}},
  author={Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```
