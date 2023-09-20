from pathlib import Path
import argparse
from typing import Any, Callable, Generator
import yaml

RawConfig = dict[str, Any]


Dummy = Generator[tuple[RawConfig, str], None, None]

def fix_tag(tag: str) -> str:
    if '.' in tag:
        tag = tag.replace('.', 'p')
    if '_' in tag:
        tag = tag.replace('_', '-')
    return tag

def update_weight_dim(config: RawConfig) -> Dummy:
    for weight_dim in range(5):
        config['model']['weight_dim'] = weight_dim
        tag = f'weight-dim-{weight_dim}'
        yield config, tag

def update_lr(config: RawConfig) -> Dummy:
    for lr in [1, 0.1, 0.001, 0.0001]:
        config['optimizer']['learning_rate'] = lr
        tag = f'lr-{lr}'
        yield config, tag

def update_latent_len(config: RawConfig) -> Dummy:
    for latent_len in [8, 16, 32, 64]:
        config['model']['latent_len'] = latent_len
        tag = f'latent-len-{latent_len}'
        yield config, tag

def update_activation(config: RawConfig) -> Dummy:
    activation_list = [
        'ReLU',
        'LeakyReLU',
        'GELU',
        'Tanh',
        'Identity',
        'Mish',
    ]
    for activation in activation_list:
        config['model']['activation'] = activation
        tag = f'activation-{activation}'
        yield config, tag


def update_widening_factor(config: RawConfig) -> Dummy:
    for widening_factor in [1, 2, 3, 4]:
        config['model']['widening_factor'] = widening_factor
        tag = f'widening-factor-{widening_factor}'
        yield config, tag


UPDATE_FUNC_LIST = [
    update_weight_dim,
    update_lr,
    update_latent_len,
    update_activation,
    update_widening_factor,
]

UPDATE_FUNC_DICT = {each.__name__.removeprefix('update_'): each
                    for each in UPDATE_FUNC_LIST}



def run(src: Path, dst_dir: Path, update_func: Callable[[RawConfig], Dummy]):
    with open(src) as stream:
        config = yaml.safe_load(stream)

    dst_dir.mkdir()

    for new_config, tag in update_func(config):
        tag = fix_tag(tag)
        dst_name = f'{src.stem}-{tag}.yaml'
        dst = dst_dir / dst_name
        print(f'writing {dst.resolve()}')
        with open(dst, 'w') as stream:
            yaml.safe_dump(new_config, stream)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-s', '--src', required=True, type=Path, help='Help text')
    parser.add_argument('-u', '--update', default='weight_dim',
                        choices=sorted(UPDATE_FUNC_DICT.keys()),
                        type=str, help='Help text')
    args = parser.parse_args()

    update_func_name = f'update_{args.update}'
    update_func = globals()[update_func_name]

    dst_dir = args.src.parent / f'{args.src.stem}-{args.update}'

    run(
        src=args.src,
        dst_dir=dst_dir,
        # update_func=update_weight_dim
        update_func=update_func
    )

if __name__ == "__main__":
    main()
