
import torch


class MomentumSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")

        self.params = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [
            torch.zeros_like(p) for p in self.params
        ]

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise ValueError("Sparse gradients are not supported")

            v.mul_(self.momentum).add_(p.grad)
            p.add_(v, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "velocity": [v.clone() for v in self.velocity],
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        saved_velocity = state["velocity"]

        if len(saved_velocity) != len(self.params):
            raise ValueError("Optimizer state size mismatch")

        for p, saved in zip(self.params, saved_velocity):
            if p.shape != saved.shape:
                raise ValueError("Momentum buffer shape mismatch")

        self.lr = state["lr"]
        self.momentum = state["momentum"]

        for current, saved in zip(
            self.velocity, saved_velocity
        ):
            current.copy_(saved)


class Adam:
    def __init__(self, parameters, lr=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("Epsilon must be positive")

        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.steps = [0] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise ValueError("Sparse gradients are not supported")

            self.steps[i] += 1
            t = self.steps[i]

            grad = p.grad
            m = self.m[i]
            v = self.v[i]

            m.mul_(self.beta1).add_(
                grad, alpha=1 - self.beta1
            )
            v.mul_(self.beta2).addcmul_(
                grad, grad, value=1 - self.beta2
            )

            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            update = m_hat / (v_hat.sqrt() + self.eps)
            p.add_(update, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "m": [x.clone() for x in self.m],
            "v": [x.clone() for x in self.v],
            "steps": self.steps.copy(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if any(
            len(state[key]) != len(self.params)
            for key in ("m", "v", "steps")
        ):
            raise ValueError("Optimizer state size mismatch")

        for p, m, v in zip(
            self.params, state["m"], state["v"]
        ):
            if p.shape != m.shape or p.shape != v.shape:
                raise ValueError("Optimizer state shape mismatch")

        self.lr = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]

        self.steps = state["steps"].copy()

        for current, saved in zip(self.m, state["m"]):
            current.copy_(saved)

        for current, saved in zip(self.v, state["v"]):
            current.copy_(saved)

class NAdam:
    def __init__(self, parameters, lr=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("Epsilon must be positive")

        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.steps = [0] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise ValueError("Sparse gradients are not supported")

            self.steps[i] += 1
            t = self.steps[i]
            grad = p.grad

            m = self.m[i]
            v = self.v[i]

            m.mul_(self.beta1).add_(
                grad, alpha=1 - self.beta1
            )
            v.mul_(self.beta2).addcmul_(
                grad, grad, value=1 - self.beta2
            )

            correction1 = 1 - self.beta1 ** t
            correction2 = 1 - self.beta2 ** t

            m_hat = m / correction1
            v_hat = v / correction2

            m_nesterov = (
                self.beta1 * m_hat
                + (1 - self.beta1) * grad / correction1
            )

            update = m_nesterov / (v_hat.sqrt() + self.eps)
            p.add_(update, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "m": [x.clone() for x in self.m],
            "v": [x.clone() for x in self.v],
            "steps": self.steps.copy(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if any(
            len(state[key]) != len(self.params)
            for key in ("m", "v", "steps")
        ):
            raise ValueError("Optimizer state size mismatch")

        for p, m, v in zip(
            self.params, state["m"], state["v"]
        ):
            if p.shape != m.shape or p.shape != v.shape:
                raise ValueError("Optimizer state shape mismatch")

        self.lr = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]
        self.steps = state["steps"].copy()

        for current, saved in zip(self.m, state["m"]):
            current.copy_(saved)

        for current, saved in zip(self.v, state["v"]):
            current.copy_(saved)
