"""
Authors     : Eko Widianto
Tests       : option_types.test_option_models.test_black_76_model
Description : 
"""

from .abstract_option_model import AbstractOptionModel


class AbstractBlack76Model(AbstractOptionModel):
    pass  # TODO: add __init__


class VanillaBlack76Model(AbstractBlack76Model):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalCashOrNothingBlack76Model(AbstractBlack76Model):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalAssetOrNothingBlack76Model(AbstractBlack76Model):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae
