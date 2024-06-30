"""
Module algorithm.py
"""
import logging

import pymc

import src.elements.points as pi


class Algorithm:
    """
    For model building
    """

    def __init__(self):
        """
        Constructor
        """

        logging.basicConfig(level=logging.INFO,
                            format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.__logger = logging.getLogger(__name__)

    def exc(self, data: pi.Points) -> pymc.Model:
        """

        :param data: For details, refer to src/element/points
        :return:
            model: a PyMC Model
        """

        with pymc.Model() as model:

            # Priors
            # A prior distribution for the
            #   residuals/noise/errors, sigma
            #   intercept, c
            #   gradient, m
            sigma = pymc.HalfCauchy(name='sigma', beta=10)
            intercept = pymc.Normal(name='intercept', mu=0, sigma=20)
            gradient = pymc.Normal(name='gradient', mu=0, sigma=20)

            # Hypothesis
            # noinspection PyUnresolvedReferences
            regression = pymc.Deterministic('regression', intercept + gradient * data.independent)

            # Define likelihood
            likelihood = pymc.Normal(name='y', mu=regression, sigma=sigma, observed=data.dependent)
            self.__logger.info('Likelihood: %s', likelihood)

        return model
