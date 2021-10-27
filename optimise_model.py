from torch.nn.utils import clip_grad_norm_

from utils.network import copy_gradients


def optimise_model(shared_model, local_model, loss, optimiser, lock):
    # Compute gradients
    loss.backward()

    clip_grad_norm_(local_model.parameters(), 10)

    # ---- The critical section begins ----
    lock.acquire()

    # Copy gradients to shared_model
    copy_gradients(shared_model, local_model)
    # Take a gradient step
    optimiser.step()

    lock.release()
    # ---- The critical section ends ----

    # Empty local model's gradients
    local_model.zero_grad()
