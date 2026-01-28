import torch
import torch.nn as nn

from .base import BaseModel


class RepresentationMixer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, 4, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        batch_size = reps.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, reps], dim=1)
        tokens = tokens + self.positional_embedding
        mixed = self.encoder(tokens)
        return mixed[:, 0]


class ResNet18Mixer(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        mixer: nn.Module,
        classifier: nn.Module,
        preprocessing: nn.Module,
        attribute_sizes,
        mixer_loss_weight: float = 1.0,
        mixer_detach_target: bool = False,
        use_mixer_classifier: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.mixer = mixer
        self.classifier = classifier
        self.preprocessing = preprocessing
        self.attribute_sizes = attribute_sizes
        self.mixer_loss_weight = mixer_loss_weight
        self.mixer_detach_target = mixer_detach_target
        self.use_mixer_classifier = use_mixer_classifier
        self.mixer_loss_fn = nn.MSELoss()
        self._logged_metrics = self._logged_metrics + ["mixer_loss", "total_loss"]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.preprocessing is not None:
            with torch.no_grad():
                x = self.preprocessing(x)
        return self.encoder(x)

    def _split_logits(self, logits: torch.Tensor):
        outputs = []
        offset = 0
        for size in self.attribute_sizes:
            outputs.append(logits[:, offset : offset + size])
            offset += size
        return outputs

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            x = x[:, -1]
        reps = self._encode(x)
        logits = self.classifier(reps)
        if self.objective == "classification":
            return self._split_logits(logits)
        return logits

    def _classifier_outputs(self, reps: torch.Tensor):
        logits = self.classifier(reps)
        return self._split_logits(logits)

    def _compute_classification(self, reps: torch.Tensor, targets: torch.Tensor):
        logits = self._classifier_outputs(reps)
        if targets is not None and targets.dim() > 2:
            targets = targets.reshape(-1, targets.shape[-1])
        loss, attr_loss = self.loss_fn(logits, targets)
        metrics, attr_metrics = self._compute_metrics(logits, targets)
        log_dict = self._compose_logging_dict(loss, attr_loss, metrics, attr_metrics)
        return loss, log_dict

    def _compute_losses(self, x, y):
        if x.dim() == 5:
            batch_size, num_views = x.shape[:2]
            x_flat = x.reshape(batch_size * num_views, *x.shape[2:])
            reps_flat = self._encode(x_flat)
            reps = reps_flat.view(batch_size, num_views, -1)
            cls_loss, log_dict = self._compute_classification(reps_flat, y)
            mixer_loss = torch.tensor(0.0, device=reps.device)
            if self.mixer is not None and num_views >= 4:
                mixer_inputs = reps[:, :3, :]
                target_rep = reps[:, 3, :]
                if self.mixer_detach_target:
                    target_rep = target_rep.detach()
                mixed_rep = self.mixer(mixer_inputs)
                mixer_loss = self.mixer_loss_fn(mixed_rep, target_rep)
                if self.use_mixer_classifier and y is not None and y.dim() > 2:
                    mixer_logits = self._classifier_outputs(mixed_rep)
                    mixer_cls_loss, _ = self.loss_fn(mixer_logits, y[:, 3, :])
                    mixer_loss = mixer_loss + mixer_cls_loss
            return cls_loss, mixer_loss, log_dict
        reps_flat = self._encode(x)
        cls_loss, log_dict = self._compute_classification(reps_flat, y)
        mixer_loss = torch.tensor(0.0, device=reps_flat.device)
        return cls_loss, mixer_loss, log_dict

    def train_step(self, x, y, optimizer, amp_scaler=None, **kwargs):
        optimizer.zero_grad()
        if amp_scaler:
            with torch.amp.autocast("cuda"):
                cls_loss, mixer_loss, log_dict = self._compute_losses(x, y)
                total_loss = cls_loss + self.mixer_loss_weight * mixer_loss
            amp_scaler.scale(total_loss).backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=1e3
            )
            if total_grad_norm.isfinite:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            log_dict["mixer_loss"] = mixer_loss.item()
            log_dict["total_loss"] = total_loss.item()
            return log_dict
        cls_loss, mixer_loss, log_dict = self._compute_losses(x, y)
        total_loss = cls_loss + self.mixer_loss_weight * mixer_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e3)
        optimizer.step()
        log_dict["mixer_loss"] = mixer_loss.item()
        log_dict["total_loss"] = total_loss.item()
        return log_dict

    @torch.no_grad()
    def validation_step(self, x, y=None, **kwargs):
        cls_loss, mixer_loss, log_dict = self._compute_losses(x, y)
        total_loss = cls_loss + self.mixer_loss_weight * mixer_loss
        log_dict["mixer_loss"] = mixer_loss.item()
        log_dict["total_loss"] = total_loss.item()
        return log_dict
