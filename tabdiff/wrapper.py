import tempfile
import pickle
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import core TabDiff components from the package
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer


class SimpleMetadata:
    """Simple metadata compatible with SDV SingleTableMetadata structure."""

    def __init__(self, columns: Dict[str, Dict[str, str]]):
        self.columns = columns

    @classmethod
    def from_sdv_metadata(cls, sdv_metadata):
        columns = {}
        for col_name, col_info in sdv_metadata.columns.items():
            columns[col_name] = {"sdtype": col_info["sdtype"]}
        return cls(columns)


@dataclass
class TabDiffParameters:
    """Parameters for TabDiff model configuration.

    Note: All author preprocessing is disabled in this wrapper.
    Pass data already preprocessed as you intend to model it.
    """

    # Model architecture parameters
    num_layers: int = 2
    d_token: int = 4
    n_head: int = 1
    factor: int = 32
    bias: bool = True
    dim_t: int = 1024
    use_mlp: bool = True

    # Diffusion parameters
    num_timesteps: int = 50
    scheduler: str = "power_mean"
    cat_scheduler: str = "log_linear"
    noise_dist: str = "uniform_t"

    # Noise schedule parameters
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7
    eps_max: float = 1e-3
    eps_min: float = 1e-5
    rho_init: float = 7.0
    rho_offset: float = 5.0
    k_init: float = -6.0
    k_offset: float = 1.0

    # Training parameters
    steps: int = 8000
    lr: float = 0.001
    weight_decay: float = 0
    ema_decay: float = 0.997
    batch_size: int = 4096
    check_val_every: int = 2000

    # Sampling parameters
    sample_batch_size: int = 4096
    stochastic_sampler: bool = True
    second_order_correction: bool = True

    # EDM parameters
    precond: bool = True
    sigma_data: float = 1.0
    net_conditioning: str = "sigma"

    # Noise distribution parameters
    P_mean: float = -1.2
    P_std: float = 1.2

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def model_dump(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}


class SimpleTabularDataset:
    """Minimal dataset wrapper for TabDiff training."""

    def __init__(self, X_num, X_cat, num_inverse=None, int_inverse=None, cat_inverse=None):
        self.X = torch.cat((X_num, X_cat), dim=1) if X_cat.numel() > 0 else X_num
        self.d_numerical = X_num.shape[1]

        if X_cat.numel() > 0:
            self.categories = np.array([X_cat[:, i].max().item() + 1 for i in range(X_cat.shape[1])])
        else:
            self.categories = np.array([])

        # Inverse transforms are identity because we don't apply preprocessing
        def _id(x):
            return x

        self.num_inverse = num_inverse if num_inverse is not None else _id
        self.int_inverse = int_inverse if int_inverse is not None else _id
        self.cat_inverse = cat_inverse if cat_inverse is not None else _id

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]


class ModelBase:
    def __init__(self, metadata, data: pd.DataFrame, seed: int):
        # Convert SDV metadata to SimpleMetadata if needed
        if hasattr(metadata, "columns") and hasattr(list(metadata.columns.values())[0], "get"):
            self.metadata = SimpleMetadata.from_sdv_metadata(metadata)
        else:
            self.metadata = metadata
        self.data = data
        self.seed = seed


class TabDiffGeneration(ModelBase):
    """High-level training/sampling API to use TabDiff as a package.

    Usage:
        from tabdiff import TabDiffGeneration, TabDiffParameters
        model = TabDiffGeneration(metadata, df_preprocessed, seed=42, parameters=TabDiffParameters())
        model.train()
        syn = model.sample(100)
    """

    def __init__(self, metadata, data: pd.DataFrame, seed: int, parameters: TabDiffParameters):
        super().__init__(metadata=metadata, data=data, seed=seed)
        self.parameters = parameters
        self.model = None
        self.diffusion_model = None
        self.dataset_info = None
        self.trainer = None
        self.temp_dir = None
        self.device = torch.device(parameters.device)

    def _create_dataset_info(self) -> Dict[str, Any]:
        # Build num/cat indices from metadata
        numerical_columns = []
        categorical_columns = []
        for column_name, column_metadata in self.metadata.columns.items():
            if column_metadata["sdtype"] in ["numerical", "datetime"]:
                numerical_columns.append(column_name)
            elif column_metadata["sdtype"] in ["categorical", "boolean"]:
                categorical_columns.append(column_name)

        all_columns = list(self.data.columns)
        num_col_idx = list(range(len(numerical_columns)))
        cat_col_idx = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))
        target_col_idx = []

        idx_mapping = {}
        idx_name_mapping = {}
        for i, col in enumerate(all_columns):
            idx_mapping[i] = i
            idx_name_mapping[i] = col

        return {
            "task_type": "regression",
            "n_classes": None,
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "column_names": all_columns,
            "idx_mapping": idx_mapping,
            "idx_name_mapping": idx_name_mapping,
            "int_col_idx_wrt_num": [],
        }

    def _build_tensors(self):
        self.dataset_info = self._create_dataset_info()
        all_columns = list(self.data.columns)
        num_cols = [all_columns[i] for i in self.dataset_info["num_col_idx"]]
        cat_cols = [all_columns[i] for i in self.dataset_info["cat_col_idx"]]

        # Simple holdout split
        n = len(self.data)
        val_size = min(100, max(1, n // 10))
        train_df = self.data.iloc[val_size:].reset_index(drop=True)
        val_df = self.data.iloc[:val_size].reset_index(drop=True)

        # Numerical
        if num_cols:
            X_train_num_np = train_df[num_cols].to_numpy(dtype=np.float32, copy=False)
            X_val_num_np = val_df[num_cols].to_numpy(dtype=np.float32, copy=False)
        else:
            X_train_num_np = np.zeros((len(train_df), 0), dtype=np.float32)
            X_val_num_np = np.zeros((len(val_df), 0), dtype=np.float32)

        # Categorical
        X_train_cat_list, X_val_cat_list = [], []
        if cat_cols:
            for col in cat_cols:
                tr = train_df[col]
                va = val_df[col]
                if np.issubdtype(tr.dtype, np.integer):
                    tr_codes = tr.to_numpy(dtype=np.int64, copy=False)
                    va_codes = va.to_numpy(dtype=np.int64, copy=False)
                elif np.issubdtype(tr.dtype, np.floating):
                    tr_codes = np.rint(tr.to_numpy()).astype(np.int64)
                    va_codes = np.rint(va.to_numpy()).astype(np.int64)
                else:
                    all_vals = pd.concat([tr.astype(str), va.astype(str)], ignore_index=True)
                    cats = pd.Categorical(all_vals)
                    codes = cats.codes
                    tr_codes = codes[: len(tr)]
                    va_codes = codes[len(tr) :]
                X_train_cat_list.append(tr_codes)
                X_val_cat_list.append(va_codes)
            X_train_cat_np = np.stack(X_train_cat_list, axis=1)
            X_val_cat_np = np.stack(X_val_cat_list, axis=1)
        else:
            X_train_cat_np = np.zeros((len(train_df), 0), dtype=np.int64)
            X_val_cat_np = np.zeros((len(val_df), 0), dtype=np.int64)

        # Tensors
        X_train_num = torch.tensor(X_train_num_np, dtype=torch.float32)
        X_test_num = torch.tensor(X_val_num_np, dtype=torch.float32)
        X_train_cat = torch.tensor(X_train_cat_np, dtype=torch.long)
        X_test_cat = torch.tensor(X_val_cat_np, dtype=torch.long)

        # Identity inverses
        def _id(x):
            return x

        train_data = SimpleTabularDataset(X_train_num, X_train_cat, _id, _id, _id)
        val_data = SimpleTabularDataset(X_test_num, X_test_cat, _id, _id, _id)

        return train_data, val_data

    def train(self):
        """Train the TabDiff model on provided data."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.temp_dir = tempfile.mkdtemp()
        train_data, val_data = self._build_tensors()

        d_numerical, categories = train_data.d_numerical, train_data.categories

        # Build backbone and model
        backbone = UniModMLP(
            d_numerical=d_numerical,
            categories=(categories + 1).tolist() if categories.size else [],
            num_layers=self.parameters.num_layers,
            d_token=self.parameters.d_token,
            n_head=self.parameters.n_head,
            factor=self.parameters.factor,
            bias=self.parameters.bias,
            dim_t=self.parameters.dim_t,
            use_mlp=self.parameters.use_mlp,
        )
        net = Model(
            backbone,
            precond=self.parameters.precond,
            sigma_data=self.parameters.sigma_data,
            net_conditioning=self.parameters.net_conditioning,
        ).to(self.device)

        # Diffusion
        self.diffusion_model = UnifiedCtimeDiffusion(
            num_classes=categories,
            num_numerical_features=d_numerical,
            denoise_fn=net,
            y_only_model=None,
            num_timesteps=self.parameters.num_timesteps,
            scheduler=self.parameters.scheduler,
            cat_scheduler=self.parameters.cat_scheduler,
            noise_dist=self.parameters.noise_dist,
            edm_params={
                "precond": self.parameters.precond,
                "sigma_data": self.parameters.sigma_data,
                "net_conditioning": self.parameters.net_conditioning,
            },
            noise_dist_params={"P_mean": self.parameters.P_mean, "P_std": self.parameters.P_std},
            noise_schedule_params={
                "sigma_min": self.parameters.sigma_min,
                "sigma_max": self.parameters.sigma_max,
                "rho": self.parameters.rho,
                "eps_max": self.parameters.eps_max,
                "eps_min": self.parameters.eps_min,
                "rho_init": self.parameters.rho_init,
                "rho_offset": self.parameters.rho_offset,
                "k_init": self.parameters.k_init,
                "k_offset": self.parameters.k_offset,
            },
            sampler_params={
                "stochastic_sampler": self.parameters.stochastic_sampler,
                "second_order_correction": self.parameters.second_order_correction,
            },
            device=self.device,
        ).to(self.device)

        self.diffusion_model.train()

        train_loader = DataLoader(
            train_data,
            batch_size=self.parameters.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Minimal logger replacement
        class _Logger:
            def log(self, data):
                pass

            def define_metric(self, name, step_metric=None):
                pass

        self.trainer = Trainer(
            diffusion=self.diffusion_model,
            train_iter=train_loader,
            dataset=train_data,
            test_dataset=val_data,
            logger=_Logger(),
            lr=self.parameters.lr,
            weight_decay=self.parameters.weight_decay,
            steps=self.parameters.steps,
            batch_size=self.parameters.batch_size,
            check_val_every=self.parameters.check_val_every,
            sample_batch_size=self.parameters.sample_batch_size,
            model_save_path=None,
            result_save_path=None,
            device=self.device,
            ema_decay=self.parameters.ema_decay,
        )

        self.trainer.run_loop()
        self.model = _TrainedModel(
            diffusion_model=self.diffusion_model,
            trainer=self.trainer,
            dataset_info=self.dataset_info,
            data_transforms={"num_inverse": lambda x: x, "int_inverse": lambda x: x, "cat_inverse": lambda x: x},
            sample_batch_size=self.parameters.sample_batch_size,
        )

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() before .sample().")
        return self.model.sample(n)

    def get_pkl(self) -> str:
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() before saving.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="w+b") as temp_file:
            pickle.dump(self.model, temp_file)
            return temp_file.name


class _TrainedModel:
    def __init__(self, diffusion_model, trainer, dataset_info, data_transforms, sample_batch_size):
        self.diffusion_model = diffusion_model
        self.trainer = trainer
        self.dataset_info = dataset_info
        self.data_transforms = data_transforms
        self.sample_batch_size = sample_batch_size

    def sample(self, num_samples) -> pd.DataFrame:
        self.diffusion_model.eval()
        with torch.no_grad():
            syn_data = self.diffusion_model.sample_all(
                num_samples, self.sample_batch_size, keep_nan_samples=True
            )
        syn_data = syn_data.cpu().numpy()

        # Reconstruct DataFrame without target
        num_inverse = self.data_transforms["num_inverse"]
        int_inverse = self.data_transforms["int_inverse"]
        cat_inverse = self.data_transforms["cat_inverse"]
        d_numerical = len(self.dataset_info["num_col_idx"])  # may be 0

        syn_df = pd.DataFrame()
        if d_numerical > 0:
            syn_num = syn_data[:, :d_numerical]
            syn_num = int_inverse(num_inverse(syn_num)).astype(np.float32)
            num_cols = [self.dataset_info["column_names"][i] for i in self.dataset_info["num_col_idx"]]
            for i, col in enumerate(num_cols):
                syn_df[col] = syn_num[:, i]

        if len(self.dataset_info["cat_col_idx"]) > 0:
            syn_cat = syn_data[:, d_numerical:]
            syn_cat = cat_inverse(syn_cat)
            cat_cols = [self.dataset_info["column_names"][i] for i in self.dataset_info["cat_col_idx"]]
            for i, col in enumerate(cat_cols):
                syn_df[col] = syn_cat[:, i]

        return syn_df
