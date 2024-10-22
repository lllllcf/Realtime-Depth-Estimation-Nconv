# Realtime-Depth-Estimation-Nconv



## Get Started

The following codes are all run on ccv

Load useful modules

```bash
module load cuda
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
```

Create conda env

```bash
git clone https://github.com/lllllcf/Realtime-Depth-Estimation-Nconv.git
cd Realtime-Depth-Estimation-Nconv
conda env create -f environment.yml
conda activate bp3
```



## Dataset

+ You can find preprocessed KITTI and NYUv2 dataset, as well as data collected from our spot in `/oscar/data/jtompki1/cli277`.

+ Currently I use data from `/oscar/data/jtompki1/cli277/nyuv2/nyuv2`.

+ Check `./dataset/*.py` for more information related to my dataloader.



## Model



### Step1 Model

`./models/step1.py/SETP1_NCONV`

Input: `raw_depth`

Output: `estimated_depth_step1`



### Step2 Model

 `./models/step2.py/SETP2_BP_TRAIN`

Input: `estimated_depth_step1` and `rgb_image`

Output: `estimated_depth_step2`

There are `SETP2_BP_TRAIN` and `SETP2_BP_EXPORT` in `step2.py`. The former is used to train the model, and the latter is used to export the model to onnx.



### Model to Export

 `./models/step2.py/SETP2_BP_EXPORT`

There are `SETP2_BP_TRAIN` and `SETP2_BP_EXPORT` in `step2.py`. The former is used to train the model, and the latter is used to export the model to onnx.



## Training

Open `train_step1.py` and specify the following parameters, then run `python train_step1.py`

```python
output_name = "test"
num_train_epoch = 2
learning_rate = [1e-4]
weight_decay = [1e-7]
```



Open `train_step2.py` and specify the following parameters, then run `python train_step2.py`

```python
output_name = "test2"
step1_checkpoint_name = "test"
num_train_epoch = 1
learning_rate = [1e-4]
weight_decay = [1e-7]
```





## Testing

I removed the test scripts as they are outdated.



## Export to ONNX

Open `export_to_onnx.py` and specify the following parameters, then run `python export_to_onnx.py`

```python
step2_checkpoint_name = "test2"
output_onnx_name = "test"
```





## Reference

I borrow [nconv](https://arxiv.org/abs/1811.01791) code from their [implementation](https://github.com/abdo-eldesokey/nconv).

May contain some code from [bpnet](https://github.com/kundajelab/bpnet), but is not in use.