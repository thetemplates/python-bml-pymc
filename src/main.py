"""
The main module for running other classes
"""
import logging
import os
import sys

import arviz
import jax
import pymc


def main():
    """
    Entry point
    :return: None
    """

    logger = logging.getLogger(__name__)

    # Notes
    logger.info('JAX')
    logger.info(jax.devices(backend='gpu'))
    logger.info('The number of GPU devices: %s', jax.device_count(backend='gpu'))

    # Sample data
    data: pi.Points  = src.data.points.Points().exc()

    # The suggested model
    model: pymc.Model = src.model.algorithm.Algorithm().exc(data=data)

    # Setting up for inference
    sampling = src.elements.sampling.Sampling(chains=8)

    # Inference
    interface = src.model.inference.Inference(sampling=sampling)
    estimates: arviz.InferenceData = interface.exc(
        model=model, nuts_sampler='numpyro', chain_method='vectorized')
    logger.info(estimates.__dict__)

    # Delete __pycache__ directories
    src.functions.cache.Cache().exc()


if __name__ == '__main__':

    # Paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=True '
        '--xla_gpu_triton_gemm_any=True '
    )

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Classes
    import src.data.points
    import src.elements.points as pi
    import src.elements.sampling
    import src.functions.cache
    import src.model.algorithm
    import src.model.inference

    main()
