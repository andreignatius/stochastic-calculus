from abc import ABC, abstractmethod


# Option Models
class AbstractOptionModel(ABC):
    @abstractmethod
    def calculate_call_price(self) -> float:
        pass

    @abstractmethod
    def calculate_put_price(self) -> float:
        pass


class AbstractBlackScholesModel(AbstractOptionModel):
    def __init__(self, S: float, K: float, r: float, sigma: float, T: float):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        print(self.S)


class AbstractBachelierModel(AbstractOptionModel):
    # add __init__
    pass


class AbstractBlack76Model(AbstractOptionModel):
    # add __init__
    pass


class AbstractDisplacedDiffusionModel(AbstractOptionModel):
    # add __init__
    pass


# Vanilla Option Models


class VanillaBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        print(1)
        pass

    def calculate_put_price(self) -> float:
        pass


class VanillaBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class VanillaBlack76Model(AbstractBlack76Model):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class VanillaDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


# Digital Cash-Or-Nothing Option Models


class DigitalCashOrNothingBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class DigitalCashOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class DigitalCashOrNothingBlack76Model(AbstractBlack76Model):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class DigitalCashOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


# Digital Asset-Or-Nothing Models


class DigitalAssetOrNothingBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class DigitalAssetOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class DigitalAssetOrNothingBlack76Model(AbstractBlack76Model):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


class DigitalAssetOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        pass

    def calculate_put_price(self) -> float:
        pass


# Option Types
class AbstractOption(ABC):
    @abstractmethod
    def black_scholes_model(self) -> AbstractBlackScholesModel:
        pass

    @abstractmethod
    def bachelier_model(self) -> AbstractBachelierModel:
        pass

    @abstractmethod
    def black_76_model(self) -> AbstractBlack76Model:
        pass

    @abstractmethod
    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        pass


class VanillaOption(AbstractOption):
    def black_scholes_model(
        self, S: float, K: float, r: float, sigma: float, T: float
    ) -> AbstractBlackScholesModel:
        return VanillaBlackScholesModel(S, K, r, sigma, T)

    def bachelier_model(self) -> AbstractBachelierModel:
        return VanillaBachelierModel()

    def black_76_model(self) -> AbstractBlack76Model:
        return VanillaBlack76Model()

    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        return VanillaDisplacedDiffusionModel()


class DigitalCashOrNothingOption(AbstractOption):
    def black_scholes_model(self) -> AbstractBlackScholesModel:
        return DigitalCashOrNothingBlackScholesModel()

    def bachelier_model(self) -> AbstractBachelierModel:
        return DigitalCashOrNothingBachelierModel()

    def black_76_model(self) -> AbstractBlack76Model:
        return DigitalCashOrNothingBlack76Model()

    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        return DigitalCashOrNothingBlack76Model()


class DigitalAssetOrNothingOption(AbstractOption):
    def black_scholes_model(self) -> AbstractBlackScholesModel:
        return DigitalAssetOrNothingBlackScholesModel()

    def bachelier_model(self) -> AbstractBachelierModel:
        return DigitalAssetOrNothingBachelierModel()

    def black_76_model(self) -> AbstractBlack76Model:
        return DigitalAssetOrNothingBlack76Model()

    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        return DigitalAssetOrNothingDisplacedDiffusionModel()


if __name__ == "__main__":
    # Sample usage for a vanilla option using BM model
    S = 1
    K = 1
    r = 1
    sigma = 1
    T = 1

    vanilla_option = VanillaOption()
    vanilla_bm_model = vanilla_option.black_scholes_model(S, K, r, sigma, T)
    print(vanilla_bm_model.calculate_call_price())
    print(vanilla_bm_model.calculate_put_price())
