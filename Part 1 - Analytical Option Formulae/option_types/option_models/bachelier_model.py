"""
Authors     : Dylan Loo & Sarah Yin
Tests       : option_types.test_option_models.test_bachelier_model
Description : uses normal distribution of returns, allows for pricing of negative values
"""
from .abstract_option_model import AbstractOptionModel


class AbstractBachelierModel(AbstractOptionModel):
    pass  # TODO: add __init__


class VanillaBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalCashOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalAssetOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae
