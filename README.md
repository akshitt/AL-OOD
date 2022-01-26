# AL-OOD

## File descriptions
- **pneumoniamnist.npz**: Complete dataset as 6 numpy arrays (train_images, train_labels, test, val). pm_train, pm_test, pm_val are subsets with columns `images`,`labels`. just for convenience while loading.  
- **ood-1.py**: final file. Direct o/p to `.txt` file. Jupyter file was just for ease of debugging.
- **create-ood-2-img.ipynb**: for creating type-2 of OOD. View this [paper](https://arxiv.org/pdf/2007.04250.pdf) for more info. Generates *.png* images to be loaded by `torchvision.datasets.ImageFolder`.   
- **pneumonia-trust-utils.py**: Contains Custom Dataset class for pneumoniamnist. Rename to `pneumonia.py` and copy to `trust/utils/`.

## To-do
1. install [distil](https://github.com/decile-team/distil/) & [trust](https://github.com/decile-team/trust) (Preferably by cloning)
2. Edit `pneumonia.py`
    - Change paths for train,test,val npz files
    - Write Custom Dataset for new images generated. Labels for all images would be `2` (all OOD data considered as one)  
    - replace cifar10 with new dataset in `load_dataset_custom_2`.  
3. Edit `ood-1.py`:
    -  Add trust,distil to path
    -  import `load_dataset_custom_2` instead [here](https://github.com/akshitt/AL-OOD/blob/e5d3e220a7c711f3e1272e0f2e11a99415f578d8/ood-1.py#L25)
4. Run `ood-1.py` several times while altering -
    - [budget=5](https://github.com/akshitt/AL-OOD/blob/e5d3e220a7c711f3e1272e0f2e11a99415f578d8/ood-1.py#L69) (try out with 10,15,20)
    - [split_cfg['per_idc_train']](https://github.com/akshitt/AL-OOD/blob/e5d3e220a7c711f3e1272e0f2e11a99415f578d8/ood-1.py#L72) (try out with 10,15,20)
    - can use bash script for re-runs after changes.
