def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    - optimizer: optimizer for updating parameters
    - p: a variable for adjusting learning rate
    return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer
