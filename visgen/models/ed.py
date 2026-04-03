import os
import torch
import torch.nn as nn

from visgen.utils.general import plot_box, plot_codebooks_similarity
from .base import BaseModel
from .resnet_mixer import RepresentationMixer

class ExpDisentanglement(BaseModel):
    """
    Disentangled Model
    """
    def __init__(
        self,
        preprocessing: torch.nn.Module,
        feature_extractors: list,
        readouts: list,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.preprocessing = preprocessing
        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.readouts = torch.nn.ModuleList(readouts)

    def forward(self, x):
        output = []
        # Process in parallel all the disentangled paths.
        with torch.no_grad():
            x = self.preprocessing(x)
        for resnet, readout in zip(self.feature_extractors, self.readouts):
            feature = resnet(x)
            out = readout(feature)
            if isinstance(out, list):
                out = out[0]
            output.append(out)
        return output

    @torch.no_grad()
    def extract_representation(self, x):
        if x.dim() == 5:
            x = x[:, -1]
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        pieces = [extractor(x) for extractor in self.feature_extractors]
        return torch.cat(pieces, dim=1)

    @torch.no_grad()
    def plot_debug(self, x, path, **kwargs):
        self.train()
        original = plot_box(
            img=self._debug_image(x),
            path=os.path.join(path, "original.png"),
        )
        x_train = self.preprocessing(x)
        train_augm = plot_box(
            img=self._debug_image(x_train),
            path=os.path.join(path, "train_augm.png"),
        )
        self.eval()
        x_test = self.preprocessing(x)
        test_augm = plot_box(
            img=self._debug_image(x_test),
            path=os.path.join(path, "test_augm.png"),
        )
        # plot codebook similarities
        if hasattr(self.readouts[0], "codebooks"):
            intra_cb, inter_cb = plot_codebooks_similarity(
                [r.codebooks[0] for r in self.readouts],
                [r.attributes[0] for r in self.readouts],
            )
            return [original, train_augm, test_augm, intra_cb, inter_cb], [
                "original",
                "train_augm",
                "test_augm",
                "intra_codebook",
                "inter_codebook",
            ]
        else:
            return [original, train_augm, test_augm], [
                "original",
                "train_augm",
                "test_augm",
            ]


class ExpDisentanglementMixer(ExpDisentanglement):
    def __init__(
        self,
        preprocessing: torch.nn.Module,
        feature_extractors: list,
        readouts: list,
        z_dim: int,
        mixer_num_layers: int = 2,
        mixer_num_heads: int = 4,
        mixer_dropout: float = 0.0,
        mixer_rep_dim=None,
        mixer_rep_piece_dim=None,
        mixer_loss_weight: float = 1.0,
        mixer_detach_target: bool = False,
        use_mixer_classifier: bool = False,
        use_all_mixer_cases: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            preprocessing=preprocessing,
            feature_extractors=feature_extractors,
            readouts=readouts,
            *args,
            **kwargs,
        )
        self.z_dim = z_dim
        self.num_pieces = len(self.readouts)
        rep_dim = self.z_dim * self.num_pieces
        if mixer_rep_piece_dim is not None:
            mixer_piece_dim = mixer_rep_piece_dim
        elif mixer_rep_dim is not None:
            if mixer_rep_dim % self.num_pieces != 0:
                raise ValueError(
                    f"mixer_rep_dim ({mixer_rep_dim}) must be divisible by number of pieces ({self.num_pieces})."
                )
            mixer_piece_dim = mixer_rep_dim // self.num_pieces
        else:
            mixer_piece_dim = self.z_dim
        mixer_dim = mixer_piece_dim * self.num_pieces
        if mixer_piece_dim == self.z_dim:
            self.mixer_piece_projections = nn.ModuleList(
                [nn.Identity() for _ in range(self.num_pieces)]
            )
        else:
            self.mixer_piece_projections = nn.ModuleList(
                [nn.Linear(self.z_dim, mixer_piece_dim) for _ in range(self.num_pieces)]
            )
        self.mixer_output_projection = (
            nn.Identity() if mixer_dim == rep_dim else nn.Linear(mixer_dim, rep_dim)
        )
        self.mixer = RepresentationMixer(
            emb_dim=mixer_dim,
            num_layers=mixer_num_layers,
            num_heads=mixer_num_heads,
            dropout=mixer_dropout,
        )
        self.mixer_loss_fn = nn.MSELoss()
        self.mixer_loss_weight = mixer_loss_weight
        self.mixer_detach_target = mixer_detach_target
        self.use_mixer_classifier = use_mixer_classifier
        self.use_all_mixer_cases = use_all_mixer_cases
        self._logged_metrics = self._logged_metrics + ["mixer_loss", "total_loss"]

    def _unwrap_readout_output(self, out):
        return out[0] if isinstance(out, list) else out

    def _encode_representations(self, x):
        if self.preprocessing is not None:
            with torch.no_grad():
                x = self.preprocessing(x)
        pieces = [extractor(x) for extractor in self.feature_extractors]
        return torch.cat(pieces, dim=1)

    @torch.no_grad()
    def extract_representation(self, x):
        if x.dim() == 5:
            x = x[:, -1]
        return self._encode_representations(x)

    def _project_rep_pieces(self, rep):
        rep_chunks = torch.split(rep, self.z_dim, dim=-1)
        projected_chunks = [
            proj(rep_chunk)
            for proj, rep_chunk in zip(self.mixer_piece_projections, rep_chunks)
        ]
        return torch.cat(projected_chunks, dim=-1)

    def _logits_from_rep(self, rep):
        rep_chunks = torch.split(rep, self.z_dim, dim=1)
        return [
            self._unwrap_readout_output(readout(rep_chunk))
            for readout, rep_chunk in zip(self.readouts, rep_chunks)
        ]

    def _compute_classification(self, reps, y):
        logits = self._logits_from_rep(reps)
        if y is not None and y.dim() > 2:
            y = y.reshape(-1, y.shape[-1])
        loss, attr_loss = self.loss_fn(logits, y)
        metrics, attr_metrics = self._compute_metrics(logits, y)
        log_dict = self._compose_logging_dict(loss, attr_loss, metrics, attr_metrics)
        return loss, log_dict

    def _compute_losses(self, x, y):
        if x.dim() == 5:
            batch_size, num_views = x.shape[:2]
            x_flat = x.reshape(batch_size * num_views, *x.shape[2:])
            y_flat = y.reshape(batch_size * num_views, y.shape[-1])
            reps_flat = self._encode_representations(x_flat)
            cls_loss, log_dict = self._compute_classification(reps_flat, y_flat)
            reps = reps_flat.view(batch_size, num_views, -1)
            mixer_loss = torch.tensor(0.0, device=reps.device)
            if num_views >= 4:
                if self.use_all_mixer_cases:
                    case_specs = [
                        (3, [0, 1, 2]),
                        (0, [2, 1, 0]),
                        (1, [2, 3, 0]),
                        (2, [1, 0, 3]),
                    ]
                else:
                    case_specs = [(3, [0, 1, 2])]
                mixer_terms = []
                for target_idx, input_indices in case_specs:
                    mixer_inputs = reps[:, input_indices, :]
                    target_rep = reps[:, target_idx, :]
                    if self.mixer_detach_target:
                        target_rep = target_rep.detach()
                    mixed_rep = self._project_rep_pieces(mixer_inputs)
                    mixed_rep = self.mixer(mixed_rep)
                    mixed_rep = self.mixer_output_projection(mixed_rep)
                    term_loss = self.mixer_loss_fn(mixed_rep, target_rep)
                    if self.use_mixer_classifier and y is not None and y.dim() > 2:
                        mixer_logits = self._logits_from_rep(mixed_rep)
                        mixer_cls_loss, _ = self.loss_fn(mixer_logits, y[:, target_idx, :])
                        term_loss = term_loss + mixer_cls_loss
                    mixer_terms.append(term_loss)
                mixer_loss = torch.stack(mixer_terms).mean()
                if torch.isnan(mixer_loss):
                    mixer_loss = torch.zeros_like(mixer_loss)
            return cls_loss, mixer_loss, log_dict
        reps = self._encode_representations(x)
        cls_loss, log_dict = self._compute_classification(reps, y)
        mixer_loss = torch.tensor(0.0, device=reps.device)
        return cls_loss, mixer_loss, log_dict

    def forward(self, x):
        if x.dim() == 5:
            x = x[:, -1]
        reps = self._encode_representations(x)
        return self._logits_from_rep(reps)

    def train_step(self, x, y, optimizer, amp_scaler=None, **kwargs):
        step_optimizer = kwargs.get("step_optimizer", True)
        grad_accum_steps = kwargs.get("grad_accum_steps", 1)
        scaled_loss_divisor = max(1, grad_accum_steps)
        if amp_scaler:
            with torch.amp.autocast("cuda"):
                cls_loss, mixer_loss, log_dict = self._compute_losses(x, y)
                total_loss = cls_loss + self.mixer_loss_weight * mixer_loss
            amp_scaler.scale(total_loss / scaled_loss_divisor).backward()
            if step_optimizer:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1e3
                )
                if total_grad_norm.isfinite:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            log_dict["mixer_loss"] = mixer_loss.item()
            log_dict["total_loss"] = total_loss.item()
            return log_dict

        cls_loss, mixer_loss, log_dict = self._compute_losses(x, y)
        total_loss = cls_loss + self.mixer_loss_weight * mixer_loss
        (total_loss / scaled_loss_divisor).backward()
        if step_optimizer:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=1e3
            )
            if total_grad_norm.isfinite:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
