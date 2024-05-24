# Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting
Paul Engstler, Andrea Vedaldi, Iro Laina, Christian Rupprecht<br>
[Project Page](https://research.paulengstler.com/invisible-stitch) | [🤗 Demo](https://huggingface.co/spaces/paulengstler/invisible-stitch) | [Paper](https://arxiv.org/abs/2404.19758)<br>

![Method figure](.github/paper_projection_figure_stacked.jpg)

This repository contains the code to train the depth completion network, generate 3D scenes, and run the scene geometry evaluation benchmark as presented in the paper "Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting".

Abstract: *3D scene generation has quickly become a challenging new research direction, fueled by consistent improvements of 2D generative diffusion models. Most prior work in this area generates scenes by iteratively stitching newly generated frames with existing geometry.  These works often depend on pre-trained monocular depth estimators to lift the generated images into 3D, fusing them with the existing scene representation. These approaches are then often evaluated via a text metric, measuring the similarity between the generated images and a given text prompt. In this work, we make two fundamental contributions to the field of 3D scene generation. First, we note that lifting images to 3D with a monocular depth estimation model is suboptimal as it ignores the geometry of the existing scene. We thus introduce a novel depth completion model, trained via teacher distillation and self-training to learn the 3D fusion process, resulting in improved geometric coherence of the scene. Second, we introduce a new benchmarking scheme for scene generation methods that is based on ground truth geometry, and thus measures the quality of the structure of the scene.*

## Release Roadmap
- [x] Inference
- [x] High-Quality Gaussian Splatting Results
- [x] Training
- [x] Benchmark

## Getting Started
Use the `environment.yml` file to create a Conda environment with all requirements for this project.

```
conda env create -n invisible_stitch --file environment.yml
conda activate invisible_stitch
```

By default, the pre-trained checkpoint of our depth completion model will be downloaded automatically from Hugging Face. If you prefer to download it manually, find the model [here](https://huggingface.co/paulengstler/invisible-stitch) and adapt the `run.py` script(s).

## Inference

To generate a 3D scene, invoke the `run.py` script:

```shell
python3 run.py \
  --image "examples/photo-1667788000333-4e36f948de9a.jpeg" \
  --prompt "a street with traditional buildings in Kyoto, Japan" \
  --output_path "./output.ply" \
  --mode "stage"
```

For the parameter `mode`, you may provide one of the following arguments:

* `single`: Simple depth projection of the input image (no hallucation)
* `stage`: Single-step hallucination of the scene to the left and right of the input image
* `360`: Full 360-degree hallucination around the given input image

To run a 360-degree hallucination, it is recommened to use a GPU with at least 16 GB VRAM.

## Training

### Dataset Setup

To train the depth completion network from a fine-tuned ZoeDepth model, we need to generate some data first. First, we predict depth for [NYU Depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) with [Marigold](https://github.com/prs-eth/Marigold). Second, we use Marigold again to predict the depth for [Places365](http://places2.csail.mit.edu/) (original). Third, we use the depth maps for Places365 to generate inpainting masks.

Places365 can be used as-is. For NYU Depth v2, please follow the instructions [here](https://github.com/cleinc/bts/tree/master/pytorch#nyu-depvh-v2) to obtain the `sync` folder. We also need the official splits for NYU Depth v2, which can be extracted with the script `extract_official_train_test_set_from_mat.py` provided [here](https://github.com/wl-zhao/VPD/blob/main/depth/extract_official_train_test_set_from_mat.py):

```shell
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

Next, please update the paths in `predict_nyu_marigold.py`, `predict_places_marigold.py`, and `project_places_depth_preds.py`. Then run these files in this sequence. These scripts are equipped with `submitit` to be distributed across a SLURM cluster. If possible, we strongly suggest parallelizing the workload.

Finally, make sure to update the paths in `zoedepth/utils/config.py:96-175`. All done!

### Training the Model

```shell
python3 train.py -m zoedepth -d marigold_nyu \
 --batch_size=12 --debug_mode=0 \
 --save_dir="checkpoints/"
```

Consider using the `_latest.pt` as opposed to the `_best.pt` checkpoint.

## Benchmark

Our scene geometry evaluation benchmark is an approach to quantitatively compare the consistency of generated scenes. In this section, we describe how to obtain and process the datasets we used, and how to run the evaluation itself.

In this short write-up, the datasets will be placed in a `datasets` folder at the root of this repository. However, this path can be adapted easily.

### ScanNet

Obtain [ScanNet](http://www.scan-net.org) and place it into `datasets/ScanNet`. Clone the ScanNet repository to obtain the library to read the sensor data, and then run our small script to extract the individual image, depth, pose, and intrinsics files.

```
git clone https://github.com/ScanNet/ScanNet
python3 benchmark/scannet_process.py ./datasets/ScanNet --input_dir ./datasets/ScanNet/scans
```

### Hypersim

Download [Hypersim](https://github.com/apple/ml-hypersim) into `datasets/hypersim`. We additionally need the camera parameters for each scene.

```bash
wget https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv -P datasets/hypersim/
```

Then, we find image pairs with a high overlap that we will evaluate on:

```
python3 hypersim_pair_builder.py --hypersim_dir ./datasets/hypersim
```

If you have access to a cluster running SLURM, you can submit a job similar to the following (you probably need to adapt `constraint` and `partition`).

```bash
#!/bin/bash

# Parameters
#SBATCH --array=0-100%20
#SBATCH --constraint=p40
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hypersim
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=low-prio-gpu
#SBATCH --requeue

python3 hypersim_pair_builder.py --hypersim_dir ./datasets/hypersim --scene_idx ${SLURM_ARRAY_TASK_ID}
```

### Running the Evaluation

Once you have finished setting up both datasets, the evaluation may be run for each dataset.


The ScanNet evaluation script will test on the first 50 scenes and print the mean absolute error across these scenes.
```
python3 scannet_eval.py 
```

The Hypersim evaluation script will consider all image pairs that were previously computed and generate the errors for each scene as csv files. We then concatenate them and also calculate the mean absolute error.

```
python3 hypersim_eval.py --out_dir ./datasets/hypersim/results_invisible_stitch
python3 hypersim_concat_eval.py ./datasets/hypersim/results_invisible_stitch
```

Adapting these scripts to your own model is straight forward: In `scannet_eval.py`, add a new mode for your model (see lines `372-384`). In `hypersim_eval.py`, duplicate the error computation for an existing model and adapt it to your own (see lines `411-441`).

## Citation
```
@inproceedings{
    engstler2024invisible,
    title={Invisible Stitch: Generating Smooth 3D Scenes with Depth Inpainting}
    author={Paul Engstler and Andrea Vedaldi and Iro Laina and Christian Rupprecht}
    year={2024}
    booktitle={Arxiv}
}
```

## Acknowledgments

P. E., A. V., I. L., and C.R. are supported by ERC-UNION-CoG-101001212. P.E. is also supported by Meta Research. I.L. and C.R. also receive support from VisualAI EP/T028572/1.

Without the great works from previous researchers, this project would not have been possible. Thank you! Our code for the depth completion network heavily borrows from [ZoeDepth](https://github.com/isl-org/ZoeDepth). We utilize [PyTorch3D](https://pytorch3d.org) in our 3D scene generation pipeline.