# NeRF-pytorch
ssh aren10@ssh.ccv.brown.edu
interact -n 2 -t 12:00:00 -m 128g -q gpu  -X -g 2 -f geforce3090
module load cuda/11.1.1
module load cudnn/8.2.0
module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate PL3DS_Baseline
rm -r Baseline_2DCLIP_reload_rgb
git clone https://github.com/aren10/Baseline_2DCLIP_reload_rgb.git
cd Baseline_2DCLIP_reload_rgb
rm -r logs
cd nerf
scp -r /Users/jfgvl1187/Desktop/logs.zip aren10@ssh.ccv.brown.edu:/users/aren10/Baseline_2DCLIP_reload_rgb/nerf/logs.zip
unzip logs.zip
python run_nerf.py --env linux --flag train --i_weights 10000
python run_nerf.py --env linux --flag test --test_file 100000.tar
python run_nerf.py --env linux --flag video --test_file 100000.tar
ctrl + z
cd /users/aren10/data/toybox-13/0
scp /Users/jfgvl1187/Desktop/metadata.json aren10@ssh.ccv.brown.edu:/users/aren10/data/0/metadata.json

scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/nerf_query_map.png /Users/jfgvl1187/Desktop/nerf_query_map.png 
scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/gt_query_map.png /Users/jfgvl1187/Desktop/gt_query_map.png
scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/rgb_est_img.png /Users/jfgvl1187/Desktop/rgb_est_img.png
scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/rgb_gt_img.png /Users/jfgvl1187/Desktop/rgb_gt_img.png

scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/queries.mp4 /Users/jfgvl1187/Desktop/queries.mp4
scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/queries_disps.mp4 /Users/jfgvl1187/Desktop/queries_disps.mp4
scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/rgb_ests.mp4 /Users/jfgvl1187/Desktop/rgb_ests.mp4
scp aren10@ssh.ccv.brown.edu:/users/aren10/data/Nesf0_2D/rgb_disps.mp4 /Users/jfgvl1187/Desktop/rgb_disps.mp4

scp -r aren10@ssh.ccv.brown.edu:/users/aren10/Baseline_2DCLIP_reload_rgb/nerf/losses.png /Users/jfgvl1187/Desktop/losses.png

zip -r logs.zip logs
scp -r aren10@ssh.ccv.brown.edu:/users/aren10/Baseline_2DCLIP_reload_rgb/nerf/logs.zip /Users/jfgvl1187/Desktop/logs.zip
rm -r logs
rm logs.zip

myq
scancel id1 id2


[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes. Here are some videos generated by this repository (pre-trained models are provided below):

![](https://user-images.githubusercontent.com/7057863/78472232-cf374a00-7769-11ea-8871-0bc710951839.gif)
![](https://user-images.githubusercontent.com/7057863/78472235-d1010d80-7769-11ea-9be9-51365180e063.gif)

This project is a faithful PyTorch implementation of [NeRF](http://www.matthewtancik.com/nerf) that **reproduces** the results while running **1.3 times faster**. The code is based on authors' Tensorflow implementation [here](https://github.com/bmild/nerf), and has been tested to match it numerically. 

## Installation

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse
  
The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.
  
</details>

## How To Run?

### Quick Start

Download data for two example datasets: `lego` and `fern`
```
bash download_example_data.sh
```

To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```
After training for 100k iterations (~4 hours on a single 2080 Ti), you can find the following video at `logs/lego_test/lego_test_spiral_100000_rgb.mp4`.

![](https://user-images.githubusercontent.com/7057863/78473103-9353b300-7770-11ea-98ed-6ba2d877b62c.gif)

---

To train a low-res `fern` NeRF:
```
python run_nerf.py --config configs/fern.txt
```
After training for 200k iterations (~8 hours on a single 2080 Ti), you can find the following video at `logs/fern_test/fern_test_spiral_200000_rgb.mp4` and `logs/fern_test/fern_test_spiral_200000_disp.mp4`

![](https://user-images.githubusercontent.com/7057863/78473081-58ea1600-7770-11ea-92ce-2bbf6a3f9add.gif)

---

### More Datasets
To play with other scenes presented in the paper, download the data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place the downloaded dataset according to the following directory structure:
```
├── configs                                                                                                       
│   ├── ...                                                                                     
│                                                                                               
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern                                                                                                                             
│   │   └── flower  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

---

To train NeRF on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.

---

To test NeRF trained on different datasets: 

```
python run_nerf.py --config configs/{DATASET}.txt --render_only
```

replace `{DATASET}` with `trex` | `horns` | `flower` | `fortress` | `lego` | etc.


### Pre-trained Models

You can download the pre-trained models [here](https://drive.google.com/drive/folders/1jIr8dkvefrQmv737fFm2isiT6tqpbTbv). Place the downloaded directory in `./logs` in order to test it later. See the following directory structure for an example:

```
├── logs 
│   ├── fern_test
│   ├── flower_test  # downloaded logs
│   ├── trex_test    # downloaded logs
```

### Reproducibility 

Tests that ensure the results of all functions and training loop match the official implentation are contained in a different branch `reproduce`. One can check it out and run the tests:
```
git checkout reproduce
py.test
```

## Method

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution  
  
<img src='imgs/pipeline.jpg'/>

> A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views


## Citation
Kudos to the authors for their amazing results:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

However, if you find this implementation or pre-trained models helpful, please consider to cite:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
