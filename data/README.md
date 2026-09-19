# Data

This directory is not included in the code archive. To reproduce the experiments,
populate it with the datasets from Hoyle et al. (2022), using the following layout:

```
data/
├── bills-hoyle-2022/
│   ├── train.metadata.jsonl
│   └── test.metadata.jsonl
└── wikitext-103-hoyle-2022/
    ├── train.metadata.jsonl
    └── test.metadata.jsonl
```

The experiment scripts read each dataset from `data/<dataset_name>/{train,test}.metadata.jsonl`,
where `<dataset_name>` matches the `dataset_name` field in the config files under `configs/`
(`bills-hoyle-2022` or `wikitext-103-hoyle-2022`).
