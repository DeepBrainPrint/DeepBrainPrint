from abc import ABC, abstractmethod


class BetaScheduler(ABC):
    @abstractmethod
    def __call__(self, curr_epoch:int) -> float:
        pass


class ConstantBetaScheduler(BetaScheduler):
    def __init__(self, const):
        self.const = const
    def __call__(self, curr_epoch: int) -> float:
        return self.const


class StepBetaScheduler(BetaScheduler):
    def __init__(self, switch_epoch):
        self.switch_epoch = switch_epoch
    def __call__(self, curr_epoch: int) -> float:
        return 1. if curr_epoch <= self.switch_epoch else 0.


class IterBetaScheduler(BetaScheduler):
    def __init__(self, period):
        self.period = period
        self.iteration = 0
        self.flag = False
    def __call__(self, curr_epoch: int) -> float:
        self.iteration += 1
        if self.iteration >= self.period:
            self.iteration = 0
            self.flag = not self.flag
        return 1. if self.flag else 0


class LinearBetaScheduler(BetaScheduler):
    def __init__(self, epochs):
        self.epochs = epochs
    def __call__(self, curr_epoch: int) -> float:
        return (1 - curr_epoch / self.epochs) if curr_epoch <= self.epochs else 0.