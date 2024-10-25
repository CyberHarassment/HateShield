# [In Construction] HateShield: Employing Multi-Modal Large Language Models for Hate Video Detection


## Structure
```
|--data                            // saved gt data, and datasets class
|  |--datasets.py                  // datasets definition for HateMM
|  |--crossval_gt.pkl              // gt file used for cross-validation
|  |--eval_gt.pkl                  // gt file used for evaluation, only has different structure with crossval_gt
|--preprocess                      // preprocessing
|  |--get_crossval_gt.py           // get the crossval_gt file from annotation file
|  |--get_eval_gt.py               // get the eval_gt file from crossval_gt
|  |--hatemm_clipclap.py           // extract features with CLIP+CLAP
|  |--hatemm_languagebind.py       // extract features with LanguageBind
|--test_clipclap.py                // testing HateShield with CLIP+CLAP features
|--test_languagebind.py            // testing HateShield with LanguageBind features
|--eval.py                         // evaluation
