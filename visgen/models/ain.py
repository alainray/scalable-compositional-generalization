import torch 
from torch import nn

from .resnet import ResNet, BasicBlock
from .resnet_mixer import RepresentationMixer


class SplitResNet18(ResNet):
    def __init__(
        self,
        split_layers=1,
        exit_reg=10,
        **kwargs,
    ):
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            **kwargs,
        )
        assert split_layers >= 0 and split_layers <= 4, "split layers must be between 0 and 4"

        if self.replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = self.replace_stride_with_dilation

        self.split_layers = split_layers
        self.exit_reg = exit_reg
        self.layer_planes = [2**i for i in range(6, 10)]
        self.exit_head_in_channels = [64, 64, 128, 256, 512][self.split_layers]

        split_blocks = []
        for _ in self.attribute_sizes:
            if self.maxpool == 0:
                conv1 = nn.Conv2d(
                    self.in_channels,
                    self.inplanes,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                )
                maxpool = nn.Identity()
            else:
                conv1 = nn.Conv2d(
                    self.in_channels,
                    self.inplanes,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
                maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            bn1 = self._norm_layer(self.inplanes)

            split_block = [
                conv1, bn1, self.activation, maxpool
            ]
            for i in range(self.split_layers):
                if i == 0:
                    dilate = False
                    stride = 1
                else:
                    dilate = replace_stride_with_dilation[i-1]
                    stride = 2
                
                li = self._make_layer(
                    self.block, 
                    self.layer_planes[i], 
                    self.layers[i], 
                    self.activation, 
                    stride=stride, 
                    dilate=dilate, 
                    skip_init=self.skip_init
                )
                split_block.append(li)

            if self.split_layers == 4:
                split_block.append(nn.AdaptiveAvgPool2d((1, 1)))
            
            split_blocks.append(nn.Sequential(*split_block))
            
        self.split_block = nn.ModuleList(split_blocks)
        self.exit_head = nn.Linear(self.exit_head_in_channels, sum(self.attribute_sizes))
        self.exit_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        shared_blocks = []
        for i in range(self.split_layers, 4):
            if i == 0:
                dilate = False
                stride = 1
            else:
                dilate = replace_stride_with_dilation[i-1]
                stride = 2
            
            shared_block = self._make_layer(
                self.block, 
                self.layer_planes[i], 
                self.layers[i], 
                self.activation, 
                stride=stride, 
                dilate=dilate, 
                skip_init=self.skip_init
            )
            shared_blocks.append(shared_block)
        
        if len(shared_blocks) > 0:
            shared_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.shared_blocks = nn.Sequential(*shared_blocks)

        self.conv1 = None
        self.bn1 = None
        self.maxpool = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.avgpool = None

    def _encode_split(self, x):
        if self.preprocessing is not None:
            with torch.no_grad():
                x = self.preprocessing(x)

        h = []
        for split_block in self.split_block:
            h.append(split_block(x))

        h = torch.cat(h, axis=0)
        x = self.shared_blocks(h)
        x = torch.flatten(x, 1)
        x_split = torch.split(x, x.shape[0] // len(self.attribute_sizes), dim=0)
        rep = torch.cat(x_split, dim=1)
        return x_split, rep, h

    @torch.no_grad()
    def extract_representation(self, x):
        if x.dim() == 5:
            x = x[:, -1]
        _, rep, _ = self._encode_split(x)
        return rep

    def forward(self, x, mode='test'):
        x_split, _, h = self._encode_split(x)
        
        # early exit embeddings
        h = self.exit_avgpool(h)
        h = torch.flatten(h, 1)
        if h.shape[1] < sum(self.attribute_sizes):
            h = self.exit_head(h)

        if self.objective == "classification":
            # split output into separate list per attribute
            h_split = torch.split(h, h.shape[0]//len(self.attribute_sizes), dim=0)
            logits_x, logits_h = [], []
            j = 0
            for i, n in enumerate(self.attribute_sizes):
                logits_xi = x_split[i][:, j : j + n] 
                logits_hi = h_split[i][:, j : j + n]
                logits_x.append(logits_xi)
                logits_h.append(logits_hi)
                j += n

        if mode=='train':
            return logits_x, logits_h
        
        return logits_x

    def train_step(self, x, y, optimizer, amp_scaler=None, **kwargs):
        # train step
        step_optimizer = kwargs.get("step_optimizer", True)
        grad_accum_steps = kwargs.get("grad_accum_steps", 1)
        scaled_loss_divisor = max(1, grad_accum_steps)
        if amp_scaler:
            with torch.cuda.amp.autocast():
                yp, yhp = self(x, 'train')
                # main loss
                yloss, attr_loss = self.loss_fn(yp, y)
                hloss, _ = self.loss_fn(yhp, y)
                loss = yloss + self.exit_reg * hloss
            amp_scaler.scale(loss / scaled_loss_divisor).backward()
            if step_optimizer:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1e3
                )
                # update the model
                if total_grad_norm.isfinite:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        else:
            yp = self(x)
            # main loss
            loss, attr_loss = self.loss_fn(yp, y)
            (loss / scaled_loss_divisor).backward()
            if step_optimizer:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1e3
                )
                if total_grad_norm.isfinite:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # compute metrics
        metrics, attr_metrics = self._compute_metrics(yp, y)
        # compose log dictionary
        dlog = self._compose_logging_dict(loss, attr_loss, metrics, attr_metrics)
        return dlog


class SplitResNet18Mixer(SplitResNet18):
    def __init__(
        self,
        mixer_num_layers=2,
        mixer_num_heads=4,
        mixer_dropout=0.0,
        mixer_rep_dim=None,
        mixer_rep_piece_dim=None,
        mixer_loss_weight=1.0,
        mixer_detach_target=False,
        use_mixer_classifier=False,
        use_all_mixer_cases=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        rep_dim = self.out_dim * len(self.attribute_sizes)
        num_pieces = len(self.attribute_sizes)
        if mixer_rep_piece_dim is not None:
            mixer_piece_dim = mixer_rep_piece_dim
        elif mixer_rep_dim is not None:
            if mixer_rep_dim % num_pieces != 0:
                raise ValueError(
                    f"mixer_rep_dim ({mixer_rep_dim}) must be divisible by number of pieces ({num_pieces})."
                )
            mixer_piece_dim = mixer_rep_dim // num_pieces
        else:
            mixer_piece_dim = self.out_dim
        mixer_dim = mixer_piece_dim * num_pieces
        self.task_classifiers = nn.ModuleList(
            [nn.Linear(self.out_dim, attr_size) for attr_size in self.attribute_sizes]
        )
        if mixer_piece_dim == self.out_dim:
            self.mixer_piece_projections = nn.ModuleList(
                [nn.Identity() for _ in self.attribute_sizes]
            )
        else:
            self.mixer_piece_projections = nn.ModuleList(
                [nn.Linear(self.out_dim, mixer_piece_dim) for _ in self.attribute_sizes]
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

    def _split_logits(self, x_split):
        return [classifier(xi) for classifier, xi in zip(self.task_classifiers, x_split)]

    def _split_rep_chunks(self, rep):
        return torch.split(rep, self.out_dim, dim=1)

    def _project_rep_pieces(self, rep):
        rep_chunks = torch.split(rep, self.out_dim, dim=-1)
        projected_chunks = [
            proj(rep_chunk)
            for proj, rep_chunk in zip(self.mixer_piece_projections, rep_chunks)
        ]
        return torch.cat(projected_chunks, dim=-1)

    def _compute_classification(self, x_split, y):
        logits = self._split_logits(x_split)
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
            x_split_flat, reps_flat, _ = self._encode_split(x_flat)
            cls_loss, log_dict = self._compute_classification(x_split_flat, y_flat)
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
                        mixer_logits = self._split_logits(self._split_rep_chunks(mixed_rep))
                        mixer_cls_loss, _ = self.loss_fn(mixer_logits, y[:, target_idx, :])
                        term_loss = term_loss + mixer_cls_loss
                    if torch.isfinite(term_loss):
                        mixer_terms.append(term_loss)
                if mixer_terms:
                    mixer_loss = torch.stack(mixer_terms).mean()
                    if not torch.isfinite(mixer_loss):
                        mixer_loss = torch.zeros_like(mixer_loss)
            return cls_loss, mixer_loss, log_dict

        x_split, reps, _ = self._encode_split(x)
        cls_loss, log_dict = self._compute_classification(x_split, y)
        mixer_loss = torch.tensor(0.0, device=reps.device)
        return cls_loss, mixer_loss, log_dict

    def forward(self, x):
        if x.dim() == 5:
            x = x[:, -1]
        x_split, _, _ = self._encode_split(x)
        return self._split_logits(x_split)

    def train_step(self, x, y, optimizer, amp_scaler=None, **kwargs):
        step_optimizer = kwargs.get("step_optimizer", True)
        grad_accum_steps = kwargs.get("grad_accum_steps", 1)
        scaled_loss_divisor = max(1, grad_accum_steps)
        if amp_scaler:
            with torch.amp.autocast("cuda"):
                cls_loss, mixer_loss, log_dict = self._compute_losses(x, y)
                total_loss = cls_loss + self.mixer_loss_weight * mixer_loss
            if torch.isfinite(total_loss):
                amp_scaler.scale(total_loss / scaled_loss_divisor).backward()
                if step_optimizer:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=1e3
                    )
                    if total_grad_norm.isfinite:
                        amp_scaler.step(optimizer)
                        amp_scaler.update()
            if step_optimizer:
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
