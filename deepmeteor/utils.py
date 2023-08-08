import vector


class MissingET(vector.MomentumNumpy2D):

    @classmethod
    def from_array(cls, array, cylindrical):
        component_list = ['pt', 'phi'] if cylindrical else ['px', 'py']
        return cls({component: array[:, idx]
                    for idx, component in enumerate(component_list)})

    @classmethod
    def from_tensor(cls, tensor, cylindrical: bool = False):
        array = tensor.detach().cpu().numpy()
        return cls.from_array(array, cylindrical)

    @classmethod
    def from_tree(cls, tree, algo: str):
        data = tree.arrays([f'{algo}_pt', f'{algo}_phi'], library='np')
        data = {key.removeprefix(f'{algo}_'): value
                for key, value in data.items()}
        return cls(data)

    def __getitem__(self, key):
        return getattr(self, key)


def compute_residual(lhs: MissingET, rhs: MissingET, component: str):
    if component == 'phi':
        residual = lhs.deltaphi(rhs)
    else:
        residual = lhs[component] - rhs[component]
    return residual
