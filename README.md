# End-to-End Neural Discourse Deixis Resolution in Dialogue

## Overview

This repository stores code for our paper _Neural Anaphora Resolution in Dialogue Revisited_. If you use our
code, please consider citing our paper.

```
@inproceedings{li-etal-2022-neural-anaphora,
    title = "Neural Anaphora Resolution in Dialogue Revisited",
    author = "Li, Shengjie  and
      Kobayashi, Hideo  and
      Ng, Vincent",
    booktitle = "Proceedings of the CODI-CRAC 2022 Shared Task on Anaphora, Bridging, and Discourse Deixis in Dialogue",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.codi-crac.4",
    pages = "32--47",
    abstract = "We present the systems that we developed for all three tracks of the CODI-CRAC 2022 shared task, namely the anaphora resolution track, the bridging resolution track, and the discourse deixis resolution track. Combining an effective encoding of the input using the SpanBERT$_{\text{Large}}$ encoder with an extensive hyperparameter search process, our systems achieved the highest scores in all phases of all three tracks.",
}
```

Some code in this repository was borrowed
from [Xu and Choi's implementation of coref-hoi](https://github.com/lxucs/coref-hoi), [https://github.com/juntaoy/universal-anaphora-scorer](https://github.com/juntaoy/universal-anaphora-scorer), and [https://github.com/juntaoy/codi-crac2022_scripts](https://github.com/juntaoy/codi-crac2022_scripts). Please check out their repositories
too.

## Directory structure
```
ar/   # stores code for our coreference resolution system
dd/   # stores code for our discourse deixis resolution system
data/ # stores data preparation scripts for both systems
```

## Usage
1. Prepare data using `data/Convert.ipynb`. This notebook converts data files in `CONLLUA` format to `jsonlines` format.
2. Run models under `ar/` and/or `dd/`

## Questions?

If you have any questions about the code, you can create a GitHub issue or email me at [sxl180006@hlt.utdallas.edu](mailto:sxl180006@hlt.utdallas.edu).