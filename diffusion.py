import torch
import torch.nn.functional as F

def get_noise_schedule(num_timesteps=100, device="cpu"):
    """
    Calculates survival probabilities for the diffusion process.

    Creates linear schedule of betas (probability of a token being masked
    as a specific step) and converts it to a cumulative product of alphas
    (the probability that a token has survived unmasked from step 0 to t).

    Args:
        num_timesteps(int, optional): Total diffusion steps. Default 100.
        device (str or torch.device, optional): The device to store the tensor on.

    Returns:
        torch.Tensor: A 1D tensor of survival probabilities, shape (num_timesteps,).
    """

    betas = torch.linspace(0.01, 0.2, num_timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    return alphas_cumprod


def apply_absorbing_mask(x_0, t, alphas_cumprod, mask_token_id=5):
    """
    Applies the forward diffusion process (corrupting data) by replacing valid climbing holds with a 
    [MASK] token based on the current timestep.

    This uses an Absorbing Discrete Diffusion approach where tokens transition to a single absorbing state
    rather than transitioning between valid classes.
    
    Args:
        x_0 (torch.Tensor): The batch of ground-truth routes, shape (Batch, 476)
        t (torch.Tensor): A tensor of random timesteps for each item in the batch, shape (Batch,)
        alphas_cumprod (torch.Tensor): The pre-calculated survival probabilities
        mask_token_id (int): The integer representing the "MASK" class (5 in your setup)

    Returns:
        torch.Tensor: The corrupted board state (x_t), shape (Batch, 476)
    """

    batch_size, seq_len = x_0.shape
    device = x_0.device

    #  Ensure alphas_cumprod is on the same device
    alphas_cumprod = alphas_cumprod.to(device)

    alpha_bar_t = alphas_cumprod[t - 1]

    #  Reshape for broadcasting so we can compare it to our sequence
    alpha_bar_t = alpha_bar_t.unsqueeze(1)

    #  Generate random probabilities between 0 and 1 for every single hold
    rand_probs = torch.rand((batch_size, seq_len), device=device)

    #  Create a boolean mask:
    #  True if rand > alpha_bar_t (meaning we should mask it)
    #  False if rand <= alpha_bar_t (meaning we keep the original hold)
    mask = rand_probs > alpha_bar_t

    #  Create x_t by copying x_0, then applying the MASK token where mask is True
    x_t = x_0.clone()
    x_t[mask] = mask_token_id

    return x_t


def sample_discrete_diffusion(model, coords, alphas_cumprod, num_timesteps=100, mask_token_id=5):
    """
    Generates a random climbing route from scratch.

    Starts with board filled with [MASK] tokens. Iterates backwards for T to 1, asking the model
    to predict the original board, permanently unmasking a percentage of holds at each step based
    on the noise schedule.

    Args:
        model (nn.Module): The trained KilterTransformer model.
        coords (torch.Tensor): The physical (x, y) coordinates of all board nodes, shape (1, 476, 2).
        alphas_cumprod (torch.Tensor): Precomputed survival probabilities.
        num_timesteps (int, optional): Total diffusion steps. Default 100.
        mask_token_id (int, optional): Integer ID for MASK class. Default 5.

    Returns:
        torch.Tensor: The final generated route array, shape (1, 476).
    """

    device = coords.device

    #  Start with a fully masked board
    x_t = torch.full((1, 476), mask_token_id, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for t_val in reversed(range(1, num_timesteps + 1)):
            t = torch.tensor([t_val], device=device)

            #  Model predicts the full unmasked board
            logits = model(x_t, t, coords)

            #  Convert logits to probabilities
            probs = F.softmax(logits / 0.5, dim=-1)

            #  Sample a prediction based on those probabilities
            x_0_pred = torch.multinomial(probs.view(-1, 6), 1).view(1, -1)

            if t_val == 1:
                #  If it's the very last step, unmask everything that is left
                update_mask = (x_t == mask_token_id)
            else:
                alpha_bar_t = alphas_cumprod[t_val - 1]
                alpha_bar_t_minus_1 = alphas_cumprod[t_val - 2]

                #  The probability of a mask being revealed at this specific timestep
                unmask_prob = (alpha_bar_t_minus_1 - alpha_bar_t) / (1.0 - alpha_bar_t)

                rand_probs = torch.rand((1, 476), device=device)
                unmask_decision = rand_probs < unmask_prob

                update_mask = (x_t == mask_token_id) & unmask_decision

            #  Update our board state
            x_t[update_mask] = x_0_pred[update_mask]

    #  Return the fully generated route
    return x_t


def generate_constrained_batch(model, coords, alphas_cumprod, constraints, batch_size=5, num_timesteps=100, mask_token_id=5):
    """
    Generates climbing routes while adhereing to user defined constraints.

    Similar to standard sampling, but at every denoising step, it intercepts model's predicted probabilities
    and forces probabilities of user-selected holds to 1.0 for their chosen class. Also applies spacial masks, e.g.,
    preventing holds from generated above the selected finish hold.

    Args:
        model (nn.Module): The trained KilterTransformer model.
        coords (torch.Tensor): The physical (x, y) coordinates of all board nodes.
        alphas_cumprod (torch.Tensor): Precomputed survival probabilities.
        constraints (dict): User-defined holds mapping node_idx (int) -> class_id (int).
        batch_size (int, optional): Number of routes generated at once. Defaults to 5.
        num_timesteps (int, optional): Total diffusion steps. Default 100.
        mask_token_id (int, optional): Integer ID for MASK class. Default 5.

    Returns:
        torch.Tensor: The final generated route array, shape (batch_size, 476).
    """

    device = coords.device
    model.eval()

    batched_coords = coords.unsqueeze(0).expand(batch_size, -1, -1)

    #  Start with masked boards for the whole batch
    x_t = torch.full((batch_size, 476), mask_token_id, dtype=torch.long, device=device)

    has_start = any(class_id == 1 for class_id in constraints.values())
    has_finish = any(class_id == 4 for class_id in constraints.values())

    #  Apply constraints to ALL boards in the batch
    ceiling_y = None
    for node_idx, class_id in constraints.items():

        x_t[:, node_idx] = class_id

        if class_id == 4:
          ceiling_y = batched_coords[0, node_idx, 1]

    with torch.no_grad():
        for t_val in reversed(range(1, num_timesteps + 1)):
            t = torch.full((batch_size,), t_val, device=device)

            logits = model(x_t, t, batched_coords)

            probs = F.softmax(logits / 0.5, dim=-1)

            if has_start:
              probs[:, :, 1] = 0.0

            if has_finish:
              probs[:, :, 4] = 0.0

            #  Probabilities clamped
            for node_idx, class_id in constraints.items():
                probs[:, node_idx, :] = 0.0
                probs[:, node_idx, class_id] = 1.0

            if ceiling_y is not None:
              nodes_above = batched_coords[0, :, 1] < ceiling_y
              probs[:, nodes_above, :] = 0
              probs[:, nodes_above, 0] = 1

            x_0_pred = torch.multinomial(probs.view(-1, 6), 1).view(batch_size, 476)

            if t_val == 1:
                update_mask = (x_t == mask_token_id)
            else:
                alpha_bar_t = alphas_cumprod[t_val - 1]
                alpha_bar_t_minus_1 = alphas_cumprod[t_val - 2]
                unmask_prob = (alpha_bar_t_minus_1 - alpha_bar_t) / (1.0 - alpha_bar_t)

                #  Random probabilities for all boards
                rand_probs = torch.rand((batch_size, 476), device=device)
                unmask_decision = rand_probs < unmask_prob

                update_mask = (x_t == mask_token_id) & unmask_decision

            x_t[update_mask] = x_0_pred[update_mask]

    return x_t