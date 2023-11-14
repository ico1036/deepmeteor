# DeepMETEOR: Deep Learning-based MET Estimation for Online Reconstruction

## Recipes
### install
You can install python dependencies using conda.
```bash
condaa create -y -f environment.yaml
```

#### T3_KR_KNU
```bash
CONDA_OVERRIDE_CUDA=11.7 mamba create -y -f environment.yaml
```

### setup
`setup.zsh` exports two environment variables named `PROJECT_DATA_DIR` and `PROJECT_LOG_DIR`.
`PROJECT_DATA_DIR` is a directory where datasets are stored and `PROJECT_LOG_DIR` is a directory where training output files are saved.
Edit them to suit your system.
```bash
source setup.zsh
```

### preparing datasets
You can download ttbar samples using the following script.
```bash
./scripts/download-files.zsh
```

### quick start
All configuration is handled using yaml file.
```bash
python train.py ./config/transformer.yaml
```
