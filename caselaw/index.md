Content related to case law search. 

## Collection 
[Collection](collection) contains a general purpose test collection for evaluating systems for case law search. 

To create the collection: 
1. Download and extract the opinions and clusters files from https://www.courtlistener.com/api/bulk-data. The documents in our collection are those contained in the `doc-ids.txt` list. 
2. go get, build `create-docs.go`. Provide the paths to the both the folders containing the opinions and clusters files. Provide also an output path. 

We provide a mapping for mapping these documents into ElasticSearch (version 6.2.2 was used).

Topics are provided in [topics folder](collection/topics/). 

If you use this collection in your research, please cite: 
```
@inproceedings{CaselawCollection,
  title={A Test Collection for Evaluating Legal Case Law Search},
  author={Locke, Daniel and Zuccon, Guido},
  booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  pages={1261--1264},
  year={2018},
  organization={ACM}
}
```

## Assessment interface
The interface used to assess documents in the caselaw collection is made available [here](https://github.com/dan-locke/caselaw-relevance-interface). A hosted interface will be available later. 

## USSC Collection
[USSC Collection](https://github.com/ielab/ussc-caselaw-collection) contains a small collection for evaluating automatic query reduction. This collection is a subset of the documents in the main collection. Instructions for creating the collection are detailed in the repository.

If you use this collection in your research, please cite: 
```
@inproceedings{Airs,
  title={Automatic Query Generation from Legal Texts for Case Law Retrieval},
  author={Locke, Daniel and Zuccon, Guido and Scells, Harrisen},
  booktitle={Asia Information Retrieval Symposium},
  pages={181--193},
  year={2017},
  organization={Springer}
}
```
