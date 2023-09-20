import warnings

ALGO_LIST = ['gen', 'puppi', 'meteor']

COLOR_DICT = {
    'gen': 'black',
    'puppi': 'tab:green',
    'meteor': 'tab:red',
}

ALGO_LABEL_DICT = {
    'gen': 'Generated',
    'puppi': 'PUPPI',
    'meteor': 'METEOR',
    'deepmet': 'DeepMET'
}

_LABEL_DICT = {
    'px': r'$p_{x}^{miss}\ [GeV]$',
    'py': r'$p_{y}^{miss}\ [GeV]$',
    'pt': r'$p_{T}^{miss}\ [GeV]$',
    'phi': r'$\phi(\vec{p}_{T}^{miss})\ [rad]$',
    'residual_px': r'$\Delta p_{x}^{miss}\ [GeV]$',
    'residual_py': r'$\Delta p_{y}^{miss}\ [GeV]$',
    'residual_pt': r'$\Delta p_{T}^{miss}\ [GeV]$',
    'residual_phi': r'$\Delta \phi(\vec{p}_{T}^{miss})\ [rad]$',
}
COMPONENT_LIST = ['px', 'py', 'pt', 'phi']

LABEL_DICT = _LABEL_DICT.copy()

# /u/user/seyang/work/meteor/deepmeteor/deepmeteor/const.py:65: UserWarning: failed to get the label for name='loss_0_30'
#   warnings.warn(f'failed to get the label for {name=}')
# /u/user/seyang/work/meteor/deepmeteor/deepmeteor/const.py:65: UserWarning: failed to get the label for name='loss_30_60'
#   warnings.warn(f'failed to get the label for {name=}')
# /u/user/seyang/work/meteor/deepmeteor/deepmeteor/const.py:65: UserWarning: failed to get the label for name='loss_60_100'
#   warnings.warn(f'failed to get the label for {name=}')
# /u/user/seyang/work/meteor/deepmeteor/deepmeteor/const.py:65: UserWarning: failed to get the label for name='loss_100_150'
#   warnings.warn(f'failed to get the label for {name=}')
# /u/user/seyang/work/meteor/deepmeteor/deepmeteor/const.py:65: UserWarning: failed to get the label for name='loss_150_inf'

LABEL_DICT |= {
    'loss': 'Loss',
    'loss_0_30': 'Loss for $p_{T}^{miss,GEN}$ < 30 GeV',
    'loss_30_60': 'Loss for 30 GeV < $p_{T}^{miss,GEN}$ < 60 GeV',
    'loss_60_100': 'Loss for 60 GeV < $p_{T}^{miss,GEN}$ < 100 GeV',
    'loss_100_150': 'Loss for 100 GeV < $p_{T}^{miss,GEN}$ < 150 GeV',
    'loss_150_inf': 'Loss for 150 GeV < $p_{T}^{miss,GEN}$',
}

for comp in COMPONENT_LIST:
    LABEL_DICT[f'chi2_prob_{comp}'] = rf'$\chi^{{2}}$ $p$-value for ' + _LABEL_DICT[comp]
for comp in COMPONENT_LIST:
    LABEL_DICT[f'reduced_chi2_{comp}'] = rf'$\chi^{{2}}/\nu$ for ' + _LABEL_DICT[comp]

for comp in COMPONENT_LIST:
    LABEL_DICT[f'reduced_chi2_{comp}_ratio'] = rf'$(\chi^{{2}}/\nu)^{{DL}} / (\chi^{{2}}/\nu)^{{PUPPI}}$ for ' + _LABEL_DICT[comp]


LABEL_DICT |=  {
    # momentum compoents
    'residual_px_mean': r'$\mu(\Delta p_{x}^{miss,DL})\ [GeV]$',
    'residual_py_mean': r'$\mu(\Delta p_{y}^{miss,DL})\ [GeV]$',
    'residual_pt_mean': r'$\mu(\Delta p_{T}^{miss,DL})\ [GeV]$',
    'residual_phi_mean': r'$\mu(\Delta \phi(\vec{p}_{T}^{miss,DL}))\ [rad]$',
    # residual std dev
    'residual_px_std': r'$\sigma(\Delta p_{x}^{miss,DL})\ [GeV]$',
    'residual_py_std': r'$\sigma(\Delta p_{y}^{miss,DL})\ [GeV]$',
    'residual_pt_std': r'$\sigma(\Delta p_{T}^{miss,DL})\ [GeV]$',
    'residual_phi_std': r'$\sigma(\Delta \phi(\vec{p}_{T}^{miss,DL}))\ [rad]$',
    # residual std dev, deep / puppi
    'residual_px_std_ratio': r'$\sigma(\Delta p_{x}^{miss})^{DL} / \sigma(\Delta p_{x}^{miss})^{PUPPI}$',
    'residual_py_std_ratio': r'$\sigma(\Delta p_{y}^{miss})^{DL} / \sigma(\Delta p_{y}^{miss})^{PUPPI}$',
    'residual_pt_std_ratio': r'$\sigma(\Delta p_{T}^{miss})^{DL} / \sigma(\Delta p_{T}^{miss})^{PUPPI}$',
    'residual_phi_std_ratio': r'$\sigma(\Delta \phi(\vec{p}_{T}^{miss}))^{DL} / \sigma(\Delta \phi(\vec{p}_{T}^{miss}))^{PUPPI}$',
}


def get_label(name: str) -> str:
    if name not in LABEL_DICT:
        warnings.warn(f'failed to get the label for {name=}')
    return LABEL_DICT.get(name, name)
