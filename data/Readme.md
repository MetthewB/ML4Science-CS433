==================================================================================================
W2S: Widefield2SIM dataset

Authors: Ruofan Zhou*, Majed El Helou*, Daniel Sage, Thierry Laroche, Arne Seitz, Sabine Süsstruk
==================================================================================================

Summary:
This dataset contains the raw images of W2S. W2S has 360 (120 FOV x 3 channels) sets of images. Each image set contains 400 shots of Widefield images, 15 (3 phases x 5 shots) shots of SIM inputs and 1 SIM reconstructed result.

The organization of the files:
|--raw.zip                           # root folder
| |--readme.txt                      # description of the dataset
| |--Image%03d/                      # folders for different FOVs, ID range from 001 to 120
| | |--wf_channel{0,1,2}.npy         # 400 shots of the FOV captured using widefield, each file contains a 400x512x512 uint16 tensor
| | |--sim_input_channel{0,1,2}.npy  # SIM inputs for getting the SIM reconstructed results, each file contains a 1536x2560 tensor
| | |--sim_channel{0,1,2}.npy        # SIM reconstructed results of the 3 channels, each file contains a 1024x1024 tensor

Note:
- FOV IDs from 001 to 080 serve as training set, FOV IDs from 081 to 120 serve as test set
- If you are using the W2S dataset please add a reference to:
@article{zhou2020w2s,
  title={W2S: A Joint Denoising and Super-Resolution Dataset},
  author={Zhou, Ruofan and El Helou, Majed and Sage, Daniel and Laroche, Thierry and Seitz, Arne and S{\"u}sstrunk, Sabine},
  journal={arXiv preprint arXiv:2003.05961},
  year={2020}
}
