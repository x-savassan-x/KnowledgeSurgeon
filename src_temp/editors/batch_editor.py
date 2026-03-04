from enum import Enum


class BatchEditor(Enum):
    FT = 'FT'
    MEMIT = 'MEMIT'
    ALPHAEDIT = "AlphaEdit"
    SPHERE = "SPHERE"


    @staticmethod
    def is_batchable_method(alg_name: str):
        return alg_name == BatchEditor.FT.value \
            or alg_name == BatchEditor.MEMIT.value \
            or alg_name == BatchEditor.ALPHAEDIT.value \
            or alg_name == BatchEditor.SPHERE.value

