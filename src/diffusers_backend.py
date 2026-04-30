import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DiffusersUNetWrapper(nn.Module):
    """Adapter exposing a diffusers UNet2DModel as model(x, t, y=None)."""

    def __init__(self, model_id="google/ddpm-cifar10-32"):
        super().__init__()
        try:
            from diffusers import UNet2DModel
        except Exception as exc:
            raise RuntimeError(
                "Diffusers backend requested, but diffusers could not be imported. "
                "Please fix/install diffusers, transformers, accelerate, and huggingface_hub versions."
            ) from exc
        self.model_id = model_id
        self.unet = UNet2DModel.from_pretrained(model_id)

    def forward(self, x, t, y=None):
        x = x.to(dtype=next(self.unet.parameters()).dtype).contiguous()
        t = t.to(device=x.device, dtype=torch.long).contiguous()
        return self.unet(x, t).sample


class DiffusersDiffuser:
    """DDPM/DDIM training and sampling wrapper for diffusers schedulers."""

    def __init__(self, model_id="google/ddpm-cifar10-32", time_steps=1000):
        try:
            from diffusers import DDIMScheduler, DDPMScheduler
        except Exception as exc:
            raise RuntimeError(
                "Diffusers backend requested, but schedulers could not be imported."
            ) from exc
        self.model_id = model_id
        self.time_steps = int(time_steps)
        self.DDIMScheduler = DDIMScheduler
        self.DDPMScheduler = DDPMScheduler
        self.noise_scheduler = self._load_scheduler(DDPMScheduler)
        self.noise_scheduler.set_timesteps(self.time_steps)

    def _load_scheduler(self, scheduler_cls):
        try:
            return scheduler_cls.from_pretrained(self.model_id)
        except Exception:
            return scheduler_cls(
                num_train_timesteps=self.time_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
            )

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l2", labels=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        predicted_noise = denoise_model(x_noisy, t, y=labels)

        if loss_type == "l1":
            return F.l1_loss(noise, predicted_noise)
        if loss_type == "huber":
            return F.smooth_l1_loss(noise, predicted_noise)
        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def sample(self, model, time_steps, image_size, batch_size=16, channels=3, labels=None,
               all_steps=False, use_ddim=False, ddim_steps=100):
        device = next(model.parameters()).device
        shape = (batch_size, channels, image_size, image_size)
        image = torch.randn(shape, device=device)

        scheduler_cls = self.DDIMScheduler if use_ddim else self.DDPMScheduler
        scheduler = self._load_scheduler(scheduler_cls)
        scheduler.set_timesteps(int(ddim_steps if use_ddim else time_steps), device=device)

        images = []
        desc = "DDIM sampling loop" if use_ddim else "sampling loop time step"
        for t in tqdm(scheduler.timesteps, desc=desc):
            t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
            predicted_noise = model(image, t_batch, y=labels)
            image = scheduler.step(predicted_noise, t, image).prev_sample
            if all_steps:
                images.append(image)
        if not all_steps:
            images.append(image)
        return images
