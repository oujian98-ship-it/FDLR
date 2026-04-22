import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import extract


# https://huggingface.co/blog/annotated-diffusion
class Diffuser:
    def __init__(self, time_steps):
        # define beta schedule
        self.betas = self.linear_beta_schedule(time_steps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def linear_beta_schedule(self, time_steps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, time_steps)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, labels=None):
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t, y=labels) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def ddim_sample(self, model, x, t, t_prev, labels=None):
        """
        DDIM sampling step (eta=0 for deterministic sampling).
        """
        # alpha_t and alpha_t_prev are alphas_cumprod
        alpha_t = extract(self.alphas_cumprod, t, x.shape)
        if t_prev >= 0:
            alpha_t_prev = extract(self.alphas_cumprod, torch.tensor([t_prev], device=x.device), x.shape)
        else:
            alpha_t_prev = torch.ones_like(alpha_t)

        # Predict noise
        predicted_noise = model(x, t, y=labels)

        # Predict x_0
        pred_x0 = (x - torch.sqrt(1. - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1. - alpha_t_prev) * predicted_noise

        # x_{t-1}
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, model, time_steps, shape, labels=None, all_steps=False):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        image = torch.randn(shape, device=device)
        images = []

        for i in tqdm(reversed(range(0, time_steps)), desc='sampling loop time step', total=time_steps):
            image = self.p_sample(model, image, torch.full((b,), i, device=device, dtype=torch.long), i, labels=labels)
            if all_steps or i == 0:
                images.append(image)
        return images

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, ddim_steps=100, labels=None, all_steps=False):
        device = next(model.parameters()).device
        b = shape[0]
        
        # total_steps is the original DDPM steps (e.g. 1000)
        total_steps = len(self.betas)
        
        # Sub-sample timesteps for DDIM
        # e.g. [0, 10, 20, ..., 990] if ddim_steps=100 and total_steps=1000
        times = torch.linspace(-1, total_steps - 1, steps=ddim_steps + 1, dtype=torch.long)
        times = list(reversed(times.tolist()))
        
        image = torch.randn(shape, device=device)
        images = []

        for i in tqdm(range(len(times) - 1), desc='DDIM sampling loop'):
            t = torch.full((b,), times[i], device=device, dtype=torch.long)
            t_prev = times[i+1]
            
            image = self.ddim_sample(model, image, t, t_prev, labels=labels)
            
            if all_steps or t_prev == -1:
                images.append(image)
        
        return images

    # @torch.no_grad()
    def sample(self, model, time_steps, image_size, batch_size=16, channels=3, labels=None, 
               all_steps=False, use_ddim=False, ddim_steps=100):
        shape = (batch_size, channels, image_size, image_size)
        if use_ddim:
            return self.ddim_sample_loop(model, shape, ddim_steps=ddim_steps, labels=labels, all_steps=all_steps)
        else:
            return self.p_sample_loop(model, time_steps, shape=shape, labels=labels, all_steps=all_steps)

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", labels=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, y=labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
