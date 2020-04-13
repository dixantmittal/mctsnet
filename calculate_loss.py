import torch as t
import torch.nn.functional as f

from device import Device


# Calculate gradient_ for a single sample
def calculate_loss(training_data, action, args):
    # find the predictions, embeddings and sampled actions
    predictions, logits, actions = training_data

    # duplicate action len(predictions) times to get loss after each simulation
    action = t.tensor([action] * len(predictions)).long().to(Device.get_device())

    predictions = t.stack(predictions)

    # Compute cross entropy loss to train differentiable parts
    loss = f.cross_entropy(predictions[-1].unsqueeze(0), action[-1].unsqueeze(0))

    loss += args.beta * t.sum(t.softmax(predictions[-1], dim=0) * t.log_softmax(predictions[-1], dim=0))

    # Compute decrease in loss after each simulation
    l_m = f.cross_entropy(predictions, action, reduction='none').clone().detach()
    r_m = l_m[:-1] - l_m[1:]

    # compute geometric sum for difference in loss
    for i in reversed(range(0, len(r_m) - 1)):
        r_m[i] = r_m[i] + args.gamma * r_m[i + 1]

    # calculate loss for tree search actions
    for l_m, logits_m, action_m in zip(r_m, logits, actions):
        action_m = t.tensor(action_m).long().to(Device.get_device())

        # find logits
        logits_m = t.stack(logits_m)

        # find negative likelihood to minimise
        negative_log_likelihood = f.cross_entropy(logits_m, action_m, reduction='sum') * l_m

        # add it to loss
        loss += negative_log_likelihood

    return loss
