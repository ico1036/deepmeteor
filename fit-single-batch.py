#!/usr/bin/env python
import sys
import torch
from torch.utils.data import DataLoader
import tqdm
from torchhep.optim import configure_optimizers
from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.transformations import Standardization
from deepmeteor.data.eventweighting.densityweight import DensityWeightHist
from deepmeteor.models.utils import find_model_config_cls
from deepmeteor.env import find_from_data_dir


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('usage: python train.py MODEL_NAME')
        sys.exit(1)
    model_name, *argv = sys.argv[1:]

    device = torch.device('cpu')

    model_config_cls = find_model_config_cls(model_name)
    model_config = model_config_cls.from_args(argv)
    model = model_config.build().to(device)
    print(model)

    data_xform = Standardization.from_dict({
        # 'gen_met_std': [60, 60],
        'gen_met_std': [1, 1],
        'puppi_cands_cont_std': [10, 10, 1, 1]
    })

    event_weighting_path = find_from_data_dir('DensityWeightHist.npz')
    event_weighting = DensityWeightHist.from_npz(event_weighting_path)

    dataset_path = str(find_from_data_dir('perfNano_TTbar_PU200.110X_set0.root'))
    dataset = MeteorDataset.from_root(
        path_list=[dataset_path],
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=0,
        entry_stop=8192)

    data_loader = DataLoader(dataset, batch_size=4096, collate_fn=MeteorDataset.collate)
    batch = next(iter(data_loader)).to(device)

    optimizer = configure_optimizers(model, learning_rate=0.1, fused=False)
    loss_fn = torch.nn.L1Loss()

    for _ in (pbar := tqdm.trange(10000)):
        optimizer.zero_grad(set_to_none=True)
        output = model.run(batch)
        loss = loss_fn(input=output, target=batch.target)
        loss.backward()
        optimizer.step()

        loss = loss.item()
        pbar.set_description(f'{loss=:.4f}')


if __name__ == "__main__":
    main()
