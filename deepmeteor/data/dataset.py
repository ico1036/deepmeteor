from typing import Callable
import numpy as np
import uproot
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import unpad_sequence
from torch.utils.data import Dataset
from deepmeteor.data.example import Example
from deepmeteor.data.example import Batch
from deepmeteor.data.transformations import DataTransformation


# FIXME
def remove_outlier(arr):
    mask = np.abs(arr) > 500
    arr[mask] = 0
    return arr


charge_encoding = {
    -999: 0,
    -1: 1,
    0: 2,
    1: 3
}
encode_charge = np.vectorize(charge_encoding.__getitem__)

pdgid_encoding = {
    -999: 0,
    11: 5,
    13: 4,
    22: 3,
    130: 2,
    211: 1
}
encode_pdgid = np.vectorize(pdgid_encoding.__getitem__)


class MeteorDataset(Dataset):

    def __init__(self,
                 examples: list[Example],
    ) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    def __add__(self, other: 'MeteorDataset') -> 'MeteorDataset':
        return self.__class__(self.examples + other.examples)

    @classmethod
    def collate(cls, examples: list[Example]) -> Batch:
        batch = {key: [each[key] for each in examples]
                 for key in Example.field_names}

        # pop out variable length arrays
        puppi_cands_cont = batch.pop('puppi_cands_cont')
        puppi_cands_pdgid = batch.pop('puppi_cands_pdgid')
        puppi_cands_charge = batch.pop('puppi_cands_charge')

        batch = {key: torch.stack(value) for key, value in batch.items()}

        batch['puppi_cands_cont'] = pad_sequence(puppi_cands_cont, batch_first=True)
        batch['puppi_cands_pdgid'] = pad_sequence(puppi_cands_pdgid, batch_first=True)
        batch['puppi_cands_charge'] = pad_sequence(puppi_cands_charge, batch_first=True)

        batch['puppi_cands_data_mask'] = batch['puppi_cands_pdgid'] != 0
        batch['puppi_cands_length'] = batch['puppi_cands_data_mask'].sum(dim=1)

        return Batch(**batch)

    ###########################################################################
    #
    ###########################################################################
    @classmethod
    def process_root(cls,
                     path: str,
                     transformation: DataTransformation | None,
                     event_weighting, # FIXME
                     max_size: int | None,
                     entry_start: int | None = None,
                     entry_stop: int | None = None,
    ) -> list[Example]:
        """
        """
        tree = uproot.open(f'{path}:Events')

        expressions = [
            'L1PuppiCands_pt',
            'L1PuppiCands_eta',
            'L1PuppiCands_phi',
            'L1PuppiCands_pdgId',
            'L1PuppiCands_charge',
            'L1PuppiCands_puppiWeight',
            'genMet_pt',
            'genMet_phi',
            'L1PuppiMet_pt',
            'L1PuppiMet_phi'
        ]

        data = tree.arrays(expressions=expressions, library='np',
                           entry_start=entry_start, entry_stop=entry_stop)

        # input continuous variables
        ## truncate
        def truncate(arr):
            return [each[:max_size] for each in arr]

        puppi_cands_pt_arr = truncate(data['L1PuppiCands_pt'])
        puppi_cands_eta_arr = truncate(data['L1PuppiCands_eta'])
        puppi_cands_phi_arr = truncate(data['L1PuppiCands_phi'])
        puppi_cands_weight_arr = truncate(data['L1PuppiCands_puppiWeight'])

        ## remove outliers
        # TODO reason?
        puppi_cands_pt_arr = [remove_outlier(each) for each in puppi_cands_pt_arr]

        ## coordinate transformation
        puppi_cands_px_arr = [pt * np.cos(phi) for pt, phi in zip(puppi_cands_pt_arr, puppi_cands_phi_arr)]
        puppi_cands_py_arr = [pt * np.sin(phi) for pt, phi in zip(puppi_cands_pt_arr, puppi_cands_phi_arr)]

        puppi_cands_cont_arr = [np.stack(each, axis=1)
                                for each in zip(puppi_cands_px_arr,
                                                puppi_cands_py_arr,
                                                puppi_cands_eta_arr,
                                                puppi_cands_weight_arr)]

        puppi_cands_cont_arr = [torch.from_numpy(each) for each in puppi_cands_cont_arr]

        # input categorical variables
        ## truncate
        puppi_cands_pdgid_arr = truncate(data['L1PuppiCands_pdgId'])
        puppi_cands_charge_arr = truncate(data['L1PuppiCands_charge'])
        ## encode
        puppi_cands_pdgid_arr = [encode_pdgid(np.abs(each))
                           for each in puppi_cands_pdgid_arr]
        puppi_cands_charge_arr = [encode_charge(each) for each in puppi_cands_charge_arr]
        ## to tensor
        puppi_cands_pdgid_arr = [torch.from_numpy(each) for each in puppi_cands_pdgid_arr]
        puppi_cands_charge_arr = [torch.from_numpy(each)
                            for each in puppi_cands_charge_arr]

        # target
        gen_met_px_arr = data['genMet_pt'] * np.cos(data['genMet_phi'])
        gen_met_py_arr = data['genMet_pt'] * np.sin(data['genMet_phi'])
        gen_met_arr = np.stack([gen_met_px_arr, gen_met_py_arr], axis=1)
        gen_met_arr = torch.from_numpy(gen_met_arr)

        # puppi_met
        puppi_met_px_arr = data['L1PuppiMet_pt'] * np.cos(data['L1PuppiMet_phi'])
        puppi_met_py_arr = data['L1PuppiMet_pt'] * np.sin(data['L1PuppiMet_phi'])
        puppi_met_arr = np.stack([puppi_met_px_arr, puppi_met_py_arr], axis=1)
        puppi_met_arr = torch.from_numpy(puppi_met_arr)

        weight = event_weighting(data['genMet_pt'])
        weight = torch.from_numpy(weight)

        if transformation is not None:
            print('preprocessing data')
            # trick to speed up the input preprocessing
            lengths = torch.as_tensor([each.size(0) for each in puppi_cands_cont_arr])
            puppi_cands_cont_arr = pad_sequence(puppi_cands_cont_arr)
            puppi_cands_cont_arr = transformation.transform_puppi_cands_cont(
                puppi_cands_cont_arr)
            puppi_cands_cont_arr = unpad_sequence(puppi_cands_cont_arr,
                                                  lengths)

            gen_met_arr = transformation.transform_gen_met(gen_met_arr)
            puppi_met_arr = transformation.transform_gen_met(puppi_met_arr)

        field_dict = {
            'puppi_cands_cont': puppi_cands_cont_arr,
            'puppi_cands_pdgid': puppi_cands_pdgid_arr,
            'puppi_cands_charge': puppi_cands_charge_arr,
            'gen_met': gen_met_arr,
            'puppi_met': puppi_met_arr,
            'weight': weight,
        }
        field_dict = {key: value for key, value in field_dict.items()}
        return [Example(*each) for each
                in zip(*[field_dict[key] for key in Example.field_names])]

    @classmethod
    def from_root(cls,
                  path_list: list[str],
                  transformation: DataTransformation | None,
                  event_weighting: Callable,
                  max_size: int | None = 100,
                  entry_start: int | None = None,
                  entry_stop: int | None = None,
    ) -> 'MeteorDataset':
        examples: list[Example] = []
        for each in path_list:
            print(f'reading {each}')
            examples += cls.process_root(
                path=each,
                transformation=transformation,
                event_weighting=event_weighting,
                max_size=max_size,
                entry_start=entry_start,
                entry_stop=entry_stop
            )
        return cls(examples)

    @classmethod
    @property
    def cont_num_features(cls) -> int:
        return 4

    @classmethod
    @property
    def pdgid_num_embeddings(cls) -> int:
        """+1 for zero padding"""
        return 5 + 1

    @classmethod
    @property
    def charge_num_embeddings(cls) -> int:
        """{-1,0,+1} and zero padding"""
        return 3 + 1

    @classmethod
    @property
    def target_num_features(cls) -> int:
        """px and py"""
        return 2
