import torch
from torch.optim.optimizer import Optimizer

class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, debug=False):

        for group in self.param_groups:

            b1, b2 = group['betas']
            lr  = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:

                    g = p.grad

                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['step']  = 0
                        state['m']     = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['v']     = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1

                    # TODO implement Adam optimizer TODO

                    step = state['step']

                    # Retrieve first and second moment vectors
                    m = state['m']
                    v = state['v']

                    # Update biased first moment estimate (m_t)
                    m.mul_(b1).add_(g, alpha=1 - b1)

                    # Update biased second moment estimate (v_t)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)

                    # Compute bias-corrected first and second moment estimates
                    m_hat = m / (1 - b1 ** step)
                    v_hat = v / (1 - b2 ** step)

                    # Update the parameters using the Adam update rule
                    p.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)

                    if debug:
                        print(f"Step {step}, m: {m.mean().item()}, v: {v.mean().item()}, p: {p.mean().item()}")

        return None
