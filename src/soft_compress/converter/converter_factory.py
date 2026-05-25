'''
This module contains the converter factory class for creating converter objects.
'''

from base_converter import Converter
from ls_converter import LeastSquaresConverter
from procrustes_converter import ProcrustesConverter
from bp_converter import BPConverter
from mlp_ust_converter import MLPUnsupervisedConverter
from mlp_st_converter import MLPSupervisedConverter
from random_converter import RandomConverter

converter_classes = {
    "ls": LeastSquaresConverter,
    "procrustes": ProcrustesConverter,
    "bp": BPConverter,
    "mlp_ust": MLPUnsupervisedConverter,
    "mlp_st": MLPSupervisedConverter,
    "random": RandomConverter,
}

class ConverterFactory:
    '''
    This class contains the converter factory class for creating converter objects.
    '''
    def __init__(self):
        '''
        Initialize the converter factory class.
        '''
        self.converter_classes = converter_classes

    def create_converter(self, converter_type, converter_config=None, *args, **kwargs):
        key = str(converter_type).strip().lower()
        if key not in self.converter_classes:
            known = ", ".join(sorted(self.converter_classes))
            raise ValueError(
                f"Invalid converter type: {converter_type!r} (normalized: {key!r}). "
                f"Known types: {known}. "
                f"Loaded factory from: {__file__}. "
                "If 'procrustes' is missing, sync procrustes_converter.py, anchor_utils.py, "
                "and this converter_factory.py to the server."
            )
        converter_class = self.converter_classes[key]
        if converter_config is not None:
            for key, value in converter_config.items():
                kwargs[key] = value
        return converter_class(*args, **kwargs)