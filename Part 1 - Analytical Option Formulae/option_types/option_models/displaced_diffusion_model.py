from .abstract_option_model import AbstractOptionModel


class AbstractDisplacedDiffusionModel(AbstractOptionModel):
    pass  # TODO: add __init__


class VanillaDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalCashOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalAssetOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae
