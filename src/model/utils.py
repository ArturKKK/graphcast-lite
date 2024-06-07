from typing import Union
from src.config import (
    Encoders,
    AggregationEncoderConfig,
    MLPEncoderConfig,
    ProcessConfig,
    DecoderConfig,
)
from src.model.encoder import AggregationEncoder


def get_encoder_from_encoder_config(
    encoder_config: Union[AggregationEncoderConfig, MLPEncoderConfig],
    num_mesh_nodes: int,
):
    """Returns an object of the encoder model based on the encoding config provided.

    Parameters
    ----------
    encoder_config : Union[AggregationEncoderConfig, MLPEncoderConfig]
        The encoding config based on which the encoder will be loaded.
    num_mesh_nodes : int
        Number of nodes in the mesh.
    num_grid_nodes : int
        Number of nodes in the grid.
    """
    if encoder_config.encoder_name == Encoders.AGGREGATION:
        return AggregationEncoder(
            encoder_config=encoder_config,
            num_mesh_nodes=num_mesh_nodes,
        )
    else:
        raise NotImplementedError(
            f"Encoder of type {encoder_config.encoder_name} is not supported."
        )


def get_processor_from_process_config(process_config: ProcessConfig):
    """Returns an object of the process module based on the process config provided.

    Parameters
    ----------
    process_config : ProcessConfig
        The process config based on which the processor will be loaded.
    """
    pass


def get_decoder_from_decode_config(decoder_config: DecoderConfig):
    """Returns an object of the decoder module based on the decoder config provided.

    Parameters
    ----------
    decoder_config : DecoderConfig
        The decoder config based on which the decoder will be loaded.
    """
    pass
