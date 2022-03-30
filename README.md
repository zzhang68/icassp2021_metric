Companion code for the paper "An End-To-End Non-Intrusive Model for Subjective and Objective Real-World Speech Assessment Using a Multi-Task Framework", ICASSP 2021.

The dataset (e.g., human MOS scores for COSINE and VOiCES) is available [here](https://drive.google.com/drive/folders/1wIgOqnKA1U-wZQrU8eb67yQyRVOK3SnZ).

Note:
The current model takes in fixed 4s audio/speech as input, padding/truncation is needed.
Two pretrained models (on COSINE and VOiCES datasets, respectively) are provided. For different datasets, we recommend to retrain the model. 

Paper: https://ieeexplore.ieee.org/document/9414182

If you use the code in this repo, please cite the following paper:

      @inproceedings{zhang2021end,
        title={An End-To-End Non-Intrusive Model for Subjective and Objective Real-World Speech Assessment Using a Multi-Task Framework},
        author={Zhang, Zhuohuang and Vyas, Piyush and Dong, Xuan and Williamson, Donald S},
        booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={316--320},
        year={2021},
        organization={IEEE}
      }
      
If you use the dataset, please cite the following paper:

    @article{dong2020pyramid,
      title={{A pyramid recurrent network for predicting crowdsourced speech-quality ratings of real-world signals}},
      author={Dong, Xuan and Williamson, Donald S},
      booktitle={Interspeech}，
      pages={4631--4635},
      year={2020}
    }
