import copy
import torch
import torch.nn.functional as F
from .utils import select_low_loss_indices, get_ot_pair

class MACWrapper:
    def __init__(
        self,
        model,
        vae=None,
        add_weight=0.1,
        ln=True,
        model_type='full',
        one_step_weight=0.0,
        local_weight=0.0,
        local_delta=0.05,
    ):
        self.model = model
        self.vae = vae
        self.ema_model = copy.deepcopy(model).eval()
        self.ln = ln

        self.BOOTSTRAP_EVERY = 8
        self.DENOISE_TIMESTEPS = 128
        self.CLASS_DROPOUT_PROB = 0.1
        self.NUM_CLASSES = 10

        self.decay = 0.999
        self.add_weight = add_weight

        self.time_mu = 0.4
        self.time_sigma = 1.0
        self.ratio_r_not_equal_t = 0.25

        self.norm_p = 0.75
        self.norm_eps = 1e-3

        self.model_type = model_type

        # ===== NEW =====
        self.one_step_weight = one_step_weight
        self.local_weight = local_weight
        self.local_delta = local_delta

        if not (0.0 < local_delta < 1.0):
            raise ValueError("local_delta must be in (0, 1)")

    @torch.no_grad()
    def update_ema(self):
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def logit_normal_timestep_sample(self, P_mean: float, P_std: float, num_samples: int, device: torch.device) -> torch.Tensor:
        rnd_normal = torch.randn((num_samples,), device=device)
        time = torch.sigmoid(rnd_normal * P_std + P_mean)
        time = torch.clip(time, min=0.0, max=1.0)
        return time

    def sample_time_steps(self, time_sampler, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
            # Step2: Ensure t > r by sorting
            sorted_samples, _ = torch.sort(time_samples, dim=1)
            r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        elif time_sampler == "logit_normal":
            # normal_samples = torch.randn(batch_size, 2, device=device)
            # normal_samples = normal_samples * self.time_sigma + self.time_mu
            # time_samples = torch.sigmoid(normal_samples)

            t = self.logit_normal_timestep_sample(self.time_mu, self.time_sigma, batch_size, device=device)
            r = self.logit_normal_timestep_sample(-self.time_mu, self.time_sigma, batch_size, device=device)
            # Step2: Ensure t < r by sorting
            sorted_samples, _ = torch.sort(torch.stack([r, t], dim=1), dim=1)
            t, r = sorted_samples[:, 0], sorted_samples[:, 1]
        else:
            raise ValueError(f"Unknown time sampler: {time_sampler}")
    
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        t = torch.where(equal_mask, r, t)
        
        return r, t 

    def adaptive_per_sample(self, loss_per_sample):
        adp_wt = (loss_per_sample.detach() + self.norm_eps) ** self.norm_p
        return loss_per_sample / adp_wt

    def selected_mean(self, loss_per_sample, indices):
        if indices is None:
            return loss_per_sample.mean()

        if indices.numel() == 0:
            return loss_per_sample.mean() * 0.0

        return loss_per_sample[indices].mean()

    def get_loss(self, images, z0, labels, indices=None):

        self.ema_model.eval()
    
        device = images.device
        B = images.shape[0]
    
        # =====================================================
        # 1. Existing MeanFlow loss
        # =====================================================
    
        r, t = self.sample_time_steps(
            "logit_normal",
            B,
            device
        )
    
        t_full = t.view(-1, 1, 1, 1)
        r_full = r.view(-1, 1, 1, 1)
    
        x_t = (1 - t_full) * z0 + t_full * images
    
        # conditional velocity
        ut_gt = images - z0
    
        labels_dropout = torch.bernoulli(
            torch.full(labels.shape, self.CLASS_DROPOUT_PROB)
        ).to(device)
    
        labels_dropped = torch.where(
            labels_dropout.bool(),
            self.NUM_CLASSES,
            labels
        )
    
        def u_func(z, t_in, r_in):
            h = r_in - t_in
            return self.model(
                z,
                t_in,
                h,
                labels_dropped
            )
    
        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)
    
        with torch.amp.autocast("cuda", enabled=False):
    
            u_pred, dudt = torch.func.jvp(
                u_func,
                (x_t, t, r),
                (ut_gt, dtdt, drdt)
            )
    
            u_tgt = (
                ut_gt
                + (r_full - t_full) * dudt
            ).detach()
    
            mf_sq = (
                (u_pred - u_tgt) ** 2
            ).sum(dim=(1, 2, 3))
    
            mf_per_sample = self.adaptive_per_sample(
                mf_sq
            )
    
        # =====================================================
        # 2. Existing MAC weighting
        # =====================================================
    
        if indices is not None:
    
            weights = torch.ones(
                B,
                device=device
            )
    
            weights[indices] = (
                1.0 + self.add_weight
            )
    
            mf_loss = (
                mf_per_sample * weights
            ).mean()
    
        else:
    
            mf_loss = mf_per_sample.mean()
    
        total_loss = mf_loss
    
        # =====================================================
        # 3. NEW: direct 1-step loss
        # =====================================================
    
        if self.one_step_weight > 0:
    
            t0 = torch.zeros(
                B,
                device=device
            )
    
            h1 = torch.ones(
                B,
                device=device
            )
    
            u_one = self.model(
                z0,
                t0,
                h1,
                labels_dropped
            )
    
            # actual one-step generated endpoint
            x1_hat = z0 + u_one
    
            one_step_sq = (
                (x1_hat - images) ** 2
            ).sum(dim=(1, 2, 3))
    
            one_step_per_sample = (
                self.adaptive_per_sample(
                    one_step_sq
                )
            )
    
            one_step_loss = self.selected_mean(
                one_step_per_sample,
                indices
            )
    
            total_loss = (
                total_loss
                + self.one_step_weight
                * one_step_loss
            )
    
        # =====================================================
        # 4. NEW: local endpoint consistency
        # =====================================================
    
        if self.local_weight > 0:
    
            delta = self.local_delta
    
            # random local position
            s = torch.rand(
                B,
                device=device
            ) * (1.0 - delta)
    
            s_next = s + delta
    
            s_full = s.view(-1, 1, 1, 1)
            sn_full = s_next.view(-1, 1, 1, 1)
    
            # two nearby points on the SAME training path
            x_s = (
                (1.0 - s_full) * z0
                + s_full * images
            )
    
            x_sn = (
                (1.0 - sn_full) * z0
                + sn_full * images
            )
    
            # remaining horizon until t=1
            h_s = 1.0 - s
            h_sn = 1.0 - s_next
    
            u_s = self.model(
                x_s,
                s,
                h_s,
                labels_dropped
            )
    
            u_sn = self.model(
                x_sn,
                s_next,
                h_sn,
                labels_dropped
            )
    
            # predicted final endpoint from each nearby point
            x1_hat_s = (
                x_s
                + h_s.view(-1, 1, 1, 1)
                * u_s
            )
    
            x1_hat_sn = (
                x_sn
                + h_sn.view(-1, 1, 1, 1)
                * u_sn
            )
    
            local_sq = (
                (x1_hat_s - x1_hat_sn) ** 2
            ).sum(dim=(1, 2, 3))
    
            local_per_sample = (
                self.adaptive_per_sample(
                    local_sq
                )
            )
    
            local_loss = self.selected_mean(
                local_per_sample,
                indices
            )
    
            total_loss = (
                total_loss
                + self.local_weight
                * local_loss
            )
    
        return total_loss

    def forward(self, x, c, percentile, global_step):
        z0 = torch.randn_like(x)
        batch = (x, c)
        if self.model_type == 'select': 
            if self.add_weight==0:
                loss = self.get_loss(x, z0, c)
            else:
                indices_low = select_low_loss_indices(self.ema_model, batch, z0, percentile, model='meanflow')
                loss = self.get_loss(x, z0, c, indices_low)

        elif self.model_type == 'full':
            z0, x, c = get_ot_pair(self.ema_model, batch, z0, global_step, model='meanflow')
            loss = self.get_loss(x, z0, c)
 
        return loss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=16, cfg=2.0):
        """
        MeanFlow forward sampling: t=0 -> t=1.
        Each step uses (r=t_i, t=t_{i+1}), h=t-r, and updates
        z_next = z + (t-r) * u(z, t, h, cond).
        """
        device = z.device
        dtype  = z.dtype
        b = z.size(0)

        if self.vae is not None:
            self.vae.to(device)

        # Keep stepwise frames for visualization.
        images = [z if self.vae is None else self.vae.decode(z / self.vae.config.scaling_factor)[0]]

        # Time grid including both endpoints.
        time_steps = torch.linspace(0.0, 1.0, sample_steps + 1, device=device, dtype=dtype)

        for i in range(sample_steps):
            t_cur  = time_steps[i]         # Scalar tensor.
            t_next = time_steps[i + 1]
            dt     = (t_next - t_cur)      # Positive scalar tensor.

            # Build batch-shaped t/r/h tensors.
            r = torch.full((b,), t_next.item(), device=device, dtype=dtype)
            t = torch.full((b,), t_cur.item(),  device=device, dtype=dtype)
            h = r - t                       # Equal to dt.

            # Compute conditional/unconditional vector fields.
            if null_cond is not None:
                vc = self.model(z, t, h, cond)
                vu = self.model(z, t, h, null_cond)
                v  = vu + cfg * (vc - vu)   # CFG
            else:
                v  = self.model(z, t, h, cond)

            # MeanFlow forward step: z_next = z + dt * v.
            # Broadcast dt to z's shape.
            z = z + dt * v

        # Append the final visualization frame.
        if self.vae is not None:
            decoded = self.vae.decode(z / self.vae.config.scaling_factor)[0]
        else:
            decoded = z
        images.append(decoded)

        return images
