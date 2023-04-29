ssh aren10@ssh.ccv.brown.edu
interact -n 1 -t 12:00:00 -m 32g -q gpu  -X -g 1 -f geforce3090
module load cuda/11.1.1
module load cudnn/8.2.0
module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate PL3DS_Baseline


scp -r /Users/jfgvl1187/Desktop/segment-anything/main4_linux.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/main4_linux.py

scp -r /Users/jfgvl1187/Desktop/feature_stand.npy aren10@ssh.ccv.brown.edu:/users/aren10/data/datasets/seg_all_2DCLIP_gt/mic/feature_stand.npy

scp -r /Users/jfgvl1187/Desktop/segment-anything/notebooks/images/mic/train aren10@ssh.ccv.brown.edu:/users/aren10/data/datasets/seg_all_2DCLIP_gt/mic

scp -r aren10@ssh.ccv.brown.edu:/users/aren10/data/datasets/seg_all_2DCLIP_gt/mic/train/"r_52_iron\ tripod_saliency2d.png" /Users/jfgvl1187/Desktop/segment-anything/output_data/"r_52_iron\ tripod_saliency2d.png"
______

scp -r /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/train_test.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/train_test.py

scp -r /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/config.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/config.py

scp -r /Users/jfgvl1187/Desktop/segment-anything/segment_anything/Reproject_CLIP/modules.py aren10@ssh.ccv.brown.edu:/users/aren10/segment-anything/segment_anything/Reproject_CLIP/modules.py

scp -r /Users/jfgvl1187/Desktop/features aren10@ssh.ccv.brown.edu:/users/aren10/data/datasets/seg_all_2DCLIP_gt/mic/features

______
TO RUN: 
python main_nerf.py data/nerf/mic --workspace mic_rgb --fp16 --tcnn --cuda_ray

python main_nerf.py data/nerf/mic --workspace mic_clip --fp16 --tcnn --clip

python main_nerf.py data/nerf/mic --workspace mic --fp16 --tcnn --test

workspace name maters for the test since it would look for folders using that for example: mic_clip and mic_rgb for test just use mic it will use the latest checkpoints for both

--cuda_ray is only valid for RGB