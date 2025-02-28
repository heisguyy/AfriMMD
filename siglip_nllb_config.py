import copy

from transformers import SiglipVisionConfig, M2M100Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SiglipNLLBConfig(PretrainedConfig):

    model_type = "siglip_nllb"
    is_composition = True

    def __init__(self, siglip_config_dict, m2m100_config_dict, **kwargs):
        super().__init__(**kwargs)

        if m2m100_config_dict is None:
            raise ValueError("`m2m100_config_dict` can not be `None`.")

        if siglip_config_dict is None:
            raise ValueError("`siglip_config_dict` can not be `None`.")

        self.m2m100_config = M2M100Config(**m2m100_config_dict)

        self.siglip_config = SiglipVisionConfig(**siglip_config_dict)

        self.is_encoder_decoder = True

    @classmethod
    def from_siglip_nllb_configs(
        cls, siglip_config: PretrainedConfig, m2m100_config: PretrainedConfig, **kwargs
    ):
        return cls(
            siglip_config_dict=siglip_config["vision_config"].to_dict(),
            m2m100_config_dict=m2m100_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["siglip_config"] = self.siglip_config.to_dict()
        output["m2m100_config"] = self.m2m100_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
