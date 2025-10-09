import pandas as pd
import numpy as np
import torch
import json
import pickle
import os
import tempfile
from typing import Dict, Any
from dataclasses import dataclass

# TabDiff imports - using the ORIGINAL implementations
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer, split_num_cat_target, recover_data
# from tabdiff.metrics import TabMetrics
from torch.utils.data import DataLoader
# import src


class SimpleMetadata:
    """Simple metadata class to replace SDV SingleTableMetadata"""
    def __init__(self, columns: Dict[str, Dict[str, str]]):
        self.columns = columns

    @classmethod
    def from_sdv_metadata(cls, sdv_metadata):
        """Convert from SDV SingleTableMetadata to SimpleMetadata"""
        columns = {}
        for col_name, col_info in sdv_metadata.columns.items():
            columns[col_name] = {'sdtype': col_info['sdtype']}
        return cls(columns)


@dataclass
class TabDiffParameters:
    """Parameters for TabDiff model configuration"""
    # Data preprocessing parameters
    dequant_dist: str = "none"
    int_dequant_factor: float = 0.0
    
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
        """Convert to dictionary like pydantic model_dump"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }


class SimpleTabularDataset:
    """Simple dataset class to replace TabDiffDataset"""
    def __init__(self, X_num, X_cat, num_inverse=None, int_inverse=None, cat_inverse=None):
        self.X = torch.cat((X_num, X_cat), dim=1) if X_cat.numel() > 0 else X_num
        self.d_numerical = X_num.shape[1]
        
        # Calculate categories properly
        if X_cat.numel() > 0:
            self.categories = np.array([X_cat[:, i].max().item() + 1 for i in range(X_cat.shape[1])])
        else:
            self.categories = np.array([])
        
        # Store inverse transformation functions
        self.num_inverse = num_inverse if num_inverse is not None else lambda x: x
        self.int_inverse = int_inverse if int_inverse is not None else lambda x: x
        self.cat_inverse = cat_inverse if cat_inverse is not None else lambda x: x
        
    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return self.X.shape[0]


class ModelBase:
    """
    Base class for models, providing common functionalities.

    Attributes:
        model (GenerativeModel): The model instance (can be any type of model).
        metadata (SimpleMetadata): The metadata of the dataframe.
        data (pd.DataFrame): The real dataset.
        sample_number (int): The number of samples to generate.
    """

    def __init__(
        self,
        metadata,  # Can be SimpleMetadata or SDV SingleTableMetadata
        data: pd.DataFrame,
        sample_number: int,
        seed: int,
        model=None,
    ):
        self.model = model
        # Convert SDV metadata to SimpleMetadata if needed
        if hasattr(metadata, 'columns') and hasattr(list(metadata.columns.values())[0], 'get'):
            # This is SDV metadata, convert it
            self.metadata = SimpleMetadata.from_sdv_metadata(metadata)
        else:
            self.metadata = metadata
        self.data = data
        self.sample_number = sample_number
        self.seed = seed

    def generate_model(self):
        """
        Placeholder method for generating and training a model.

        Subclasses should implement this method if needed.
        """
        pass

    def generate_data(self):
        """
        Generates synthetic data using the model.

        Returns:
            pd.DataFrame: The generated synthetic data.
        """
        if self.model is None:
            self.generate_model()

        synthetic_data = self.model.sample(self.sample_number)
        return synthetic_data

    def get_pkl(self):
        """
        Saves the model to a temporary .pkl file and returns the file path.

        Returns:
            str: The path to the temporary .pkl file.
        """
        if self.model is None:
            raise ValueError("Model is not initialized")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pkl", mode="w+b"
        ) as temp_file:
            pickle.dump(self.model, temp_file)
            temp_file_name = temp_file.name

        return temp_file_name


class TabDiffGeneration(ModelBase):
    """
    TabDiff wrapper using the ORIGINAL training implementation from the paper.
    
    This class treats all columns as features to generate together, without
    requiring a specific target column designation.
    """

    def __init__(
        self,
        metadata,  # SimpleMetadata or SDV SingleTableMetadata
        data: pd.DataFrame,
        sample_number: int,
        seed: int,
        parameters: TabDiffParameters,
    ):
        super().__init__(
            metadata=metadata, data=data, sample_number=sample_number, seed=seed
        )
        self.parameters: TabDiffParameters = parameters
        self.model = None
        self.diffusion_model = None
        self.dataset_info = None
        self.data_transforms = None
        self.trainer = None
        self.temp_dir = None
        
        # Set device
        self.device = torch.device(parameters.device)
        
        # Generate synthetic data immediately after initialization
        self.synthetic_data: pd.DataFrame = self.generate_data()

    def _create_dataset_info(self) -> Dict[str, Any]:
        """
        Create dataset info for TabDiff - treating ALL columns as features to generate.
        """
        # Analyze the data and metadata to separate numerical vs categorical
        numerical_columns = []
        categorical_columns = []
        
        # Get column types from metadata
        for column_name, column_metadata in self.metadata.columns.items():
            if column_metadata['sdtype'] in ['numerical', 'datetime']:
                numerical_columns.append(column_name)
            elif column_metadata['sdtype'] in ['categorical', 'boolean']:
                categorical_columns.append(column_name)
        
        all_columns = list(self.data.columns)
        
        # Create column mappings - ALL columns are just features to generate
        num_col_idx = list(range(len(numerical_columns)))
        cat_col_idx = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))
        
        # No target column needed! We're generating ALL columns
        target_col_idx = []  # Empty - no target needed for pure generation
        
        # Create index mappings
        idx_mapping = {}
        idx_name_mapping = {}
        
        for i, col in enumerate(all_columns):
            idx_mapping[i] = i
            idx_name_mapping[i] = col
        
        # Dummy task type for TabDiff compatibility (not actually used for generation)
        task_type = 'regression'  # Doesn't matter for generation
        
        info = {
            'task_type': task_type,  # Required by TabDiff but not used for generation
            'n_classes': None,  # Not needed for generation
            'num_col_idx': num_col_idx,
            'cat_col_idx': cat_col_idx,
            'target_col_idx': target_col_idx,  # Empty!
            'column_names': all_columns,
            'idx_mapping': idx_mapping,
            'idx_name_mapping': idx_name_mapping,
            'int_col_idx_wrt_num': []  # Assume no integer columns for simplicity
        }
        
        return info

    def _prepare_data(self):
        """Prepare data in TabDiff format - treating all columns as features"""
        # Create a temporary directory to save data
        self.temp_dir = tempfile.mkdtemp()
        
        # Save data info
        self.dataset_info = self._create_dataset_info()
        info_path = os.path.join(self.temp_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(self.dataset_info, f)
        
        # Split data into numerical and categorical (NO target separation)
        all_columns = list(self.data.columns)
        num_cols = [all_columns[i] for i in self.dataset_info['num_col_idx']]
        cat_cols = [all_columns[i] for i in self.dataset_info['cat_col_idx']]
        
        # Prepare train/test split (using all data as train for generation)
        train_data = self.data.copy()
        test_data = self.data.head(100).copy()  # Small test set for compatibility
        
        # Save numerical data
        if num_cols:
            X_num_train = train_data[num_cols].values.astype(np.float32)
            X_num_test = test_data[num_cols].values.astype(np.float32)
            np.save(os.path.join(self.temp_dir, 'X_num_train.npy'), X_num_train)
            np.save(os.path.join(self.temp_dir, 'X_num_test.npy'), X_num_test)
        
        # Save categorical data
        if cat_cols:
            # Convert categorical to integer codes
            X_cat_train = train_data[cat_cols].astype('category')
            X_cat_test = test_data[cat_cols].astype('category')
            
            # Ensure same categories in train and test
            for col in cat_cols:
                combined_cats = pd.Categorical(
                    pd.concat([train_data[col], test_data[col]]).astype(str)
                ).categories
                X_cat_train[col] = pd.Categorical(train_data[col].astype(str), categories=combined_cats)
                X_cat_test[col] = pd.Categorical(test_data[col].astype(str), categories=combined_cats)
            
            X_cat_train_codes = np.column_stack([X_cat_train[col].cat.codes.values for col in cat_cols])
            X_cat_test_codes = np.column_stack([X_cat_test[col].cat.codes.values for col in cat_cols])
            
            np.save(os.path.join(self.temp_dir, 'X_cat_train.npy'), X_cat_train_codes)
            np.save(os.path.join(self.temp_dir, 'X_cat_test.npy'), X_cat_test_codes)
        
        # Create dummy target files for TabDiff compatibility (but they won't be used)
        dummy_target = np.zeros(len(train_data))
        np.save(os.path.join(self.temp_dir, 'y_train.npy'), dummy_target)
        np.save(os.path.join(self.temp_dir, 'y_test.npy'), dummy_target[:len(test_data)])

    def _custom_preprocess(self, dataset_path, y_only=False, dequant_dist='none', int_dequant_factor=0.0, task_type='regression'):
        """Custom preprocessing function that uses concat=False to avoid feature mismatch"""
        from tabdiff.utils.utils_train import preprocess
        
        # Call preprocess with concat=False to prevent target concatenation
        return preprocess(
            dataset_path=dataset_path,
            y_only=y_only,
            dequant_dist=dequant_dist,
            int_dequant_factor=int_dequant_factor,
            task_type=task_type,
            inverse=True,
            concat=False  # This is the key fix!
        )

    def generate_model(self):
        """Generate and train the TabDiff model using ORIGINAL implementation"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        print("🔥 TabDiff: Learning joint distribution of ALL your columns...")
        print("💡 Using ORIGINAL TabDiff training implementation")
        
        self._prepare_data()
        
        print("Loading and preprocessing data...")
        # Load data using TabDiff's preprocessing
        # Use our custom preprocessing to avoid concat issues
        X_num, X_cat, categories, d_numerical, num_inverse, int_inverse, cat_inverse = self._custom_preprocess(
            self.temp_dir, 
            y_only=False,
            dequant_dist=self.parameters.dequant_dist, 
            int_dequant_factor=self.parameters.int_dequant_factor,
            task_type=self.dataset_info['task_type']
        )
        
        # Create tensors directly without TabDiffDataset
        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat
        
        X_train_num = torch.tensor(X_train_num).float()
        X_test_num = torch.tensor(X_test_num).float()
        X_train_cat = torch.tensor(X_train_cat)
        X_test_cat = torch.tensor(X_test_cat)
        
        # Store preprocessed data for creating dataset objects
        self._train_data = (X_train_num, X_train_cat)
        self._val_data = (X_test_num, X_test_cat)
        
        # Create simple dataset objects with inverse transformations
        train_data = SimpleTabularDataset(X_train_num, X_train_cat, num_inverse, int_inverse, cat_inverse)
        val_data = SimpleTabularDataset(X_test_num, X_test_cat, num_inverse, int_inverse, cat_inverse)
        
        d_numerical, categories = train_data.d_numerical, train_data.categories
        
        print(f"Data structure:")
        print(f"   Numerical features: {d_numerical}")
        print(f"   Categorical features: {len(categories)} with sizes {categories}")
        print(f"   Total features: {d_numerical + len(categories)} (all will be generated!)")
        
        # Create model architecture
        backbone = UniModMLP(
            d_numerical=d_numerical,
            categories=(categories + 1).tolist(),  # add one for mask category
            num_layers=self.parameters.num_layers,
            d_token=self.parameters.d_token,
            n_head=self.parameters.n_head,
            factor=self.parameters.factor,
            bias=self.parameters.bias,
            dim_t=self.parameters.dim_t,
            use_mlp=self.parameters.use_mlp
        )
        
        model = Model(
            backbone, 
            precond=self.parameters.precond,
            sigma_data=self.parameters.sigma_data,
            net_conditioning=self.parameters.net_conditioning
        )
        model.to(self.device)
        
        # Create diffusion model
        self.diffusion_model = UnifiedCtimeDiffusion(
            num_classes=categories,
            num_numerical_features=d_numerical,
            denoise_fn=model,
            y_only_model=None,
            num_timesteps=self.parameters.num_timesteps,
            scheduler=self.parameters.scheduler,
            cat_scheduler=self.parameters.cat_scheduler,
            noise_dist=self.parameters.noise_dist,
            edm_params={
                'precond': self.parameters.precond,
                'sigma_data': self.parameters.sigma_data,
                'net_conditioning': self.parameters.net_conditioning
            },
            noise_dist_params={
                'P_mean': self.parameters.P_mean,
                'P_std': self.parameters.P_std
            },
            noise_schedule_params={
                'sigma_min': self.parameters.sigma_min,
                'sigma_max': self.parameters.sigma_max,
                'rho': self.parameters.rho,
                'eps_max': self.parameters.eps_max,
                'eps_min': self.parameters.eps_min,
                'rho_init': self.parameters.rho_init,
                'rho_offset': self.parameters.rho_offset,
                'k_init': self.parameters.k_init,
                'k_offset': self.parameters.k_offset
            },
            sampler_params={
                'stochastic_sampler': self.parameters.stochastic_sampler,
                'second_order_correction': self.parameters.second_order_correction
            },
            device=self.device
        )
        
        self.diffusion_model.to(self.device)
        self.diffusion_model.train()
        
        # Create data loader
        train_loader = DataLoader(
            train_data,
            batch_size=self.parameters.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )
        
        # Create minimal metrics for training (using dummy paths for generation)
        # We don't need real evaluation metrics for pure generation
        dummy_real_path = os.path.join(self.temp_dir, 'dummy_real.csv')
        dummy_test_path = os.path.join(self.temp_dir, 'dummy_test.csv')
        self.data.to_csv(dummy_real_path, index=False)
        self.data.head(50).to_csv(dummy_test_path, index=False)
        
        # Create ORIGINAL TabDiff trainer
        self.trainer = Trainer(
            diffusion=self.diffusion_model,
            train_iter=train_loader,
            dataset=train_data,
            test_dataset=val_data,
            logger=DummyLogger(),  # Simple logger for non-wandb training
            lr=self.parameters.lr,
            weight_decay=self.parameters.weight_decay,
            steps=self.parameters.steps,
            batch_size=self.parameters.batch_size,
            check_val_every=self.parameters.check_val_every,
            sample_batch_size=self.parameters.sample_batch_size,
            model_save_path=None,  # No saving needed for generation
            result_save_path=None,
            device=self.device,
            ema_decay=self.parameters.ema_decay
        )
        
        print("🚀 Starting training with ORIGINAL TabDiff trainer...")
        self.trainer.run_loop()
        
        # Store the trained model
        self.model = TabDiffModel(
            diffusion_model=self.diffusion_model,
            trainer=self.trainer,
            dataset_info=self.dataset_info,
            data_transforms={
                'num_inverse': train_data.num_inverse,
                'int_inverse': train_data.int_inverse,
                'cat_inverse': train_data.cat_inverse
            },
            sample_batch_size=self.parameters.sample_batch_size
        )
        
        print("✅ Training completed!")

    def generate_data(self):
        """Generate synthetic data using the trained model"""
        if self.model is None:
            self.generate_model()
        
        return self.model.sample(self.sample_number)


class TabDiffModel:
    """Wrapper for the trained TabDiff model to provide sampling interface"""
    
    def __init__(self, diffusion_model, trainer, dataset_info, data_transforms, sample_batch_size):
        self.diffusion_model = diffusion_model
        self.trainer = trainer
        self.dataset_info = dataset_info
        self.data_transforms = data_transforms
        self.sample_batch_size = sample_batch_size
    
    def sample(self, num_samples):
        """Sample synthetic data - generates ALL columns together"""
        self.diffusion_model.eval()
        
        print(f"🎲 Generating {num_samples} synthetic samples...")
        print("💫 All columns will be generated together from learned joint distribution!")
        
        # Generate raw samples using original TabDiff sampling
        with torch.no_grad():
            syn_data = self.diffusion_model.sample_all(
                num_samples, 
                self.sample_batch_size, 
                keep_nan_samples=True
            )
        
        # Convert to numpy
        syn_data = syn_data.cpu().numpy()
        
        # Since we have no target column, reconstruct differently
        if len(self.dataset_info['target_col_idx']) == 0:
            # No target column - all data is features
            num_inverse = self.data_transforms['num_inverse']
            int_inverse = self.data_transforms['int_inverse']
            cat_inverse = self.data_transforms['cat_inverse']
            
            # Split into numerical and categorical parts
            d_numerical = len(self.dataset_info['num_col_idx'])
            
            if d_numerical > 0:
                syn_num = syn_data[:, :d_numerical]
                syn_num = num_inverse(syn_num).astype(np.float32)
                syn_num = int_inverse(syn_num).astype(np.float32)
            else:
                syn_num = None
                
            if len(self.dataset_info['cat_col_idx']) > 0:
                syn_cat = syn_data[:, d_numerical:]
                syn_cat = cat_inverse(syn_cat)
            else:
                syn_cat = None
            
            # Reconstruct DataFrame
            syn_df = pd.DataFrame()
            
            # Add numerical columns
            if syn_num is not None:
                num_cols = [self.dataset_info['column_names'][i] for i in self.dataset_info['num_col_idx']]
                for i, col in enumerate(num_cols):
                    syn_df[col] = syn_num[:, i]
            
            # Add categorical columns  
            if syn_cat is not None:
                cat_cols = [self.dataset_info['column_names'][i] for i in self.dataset_info['cat_col_idx']]
                for i, col in enumerate(cat_cols):
                    syn_df[col] = syn_cat[:, i]
        else:
            # Fallback to original method if target exists
            syn_num, syn_cat, syn_target = split_num_cat_target(
                syn_data, 
                self.dataset_info, 
                self.data_transforms['num_inverse'],
                self.data_transforms['int_inverse'],
                self.data_transforms['cat_inverse']
            )
            
            # Recover DataFrame
            syn_df = recover_data(syn_num, syn_cat, syn_target, self.dataset_info)
            
            # Rename columns back to original names
            idx_name_mapping = self.dataset_info['idx_name_mapping']
            idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
            syn_df.rename(columns=idx_name_mapping, inplace=True)
        
        return syn_df


class DummyLogger:
    """Simple logger to replace wandb for non-tracking training"""
    def log(self, data):
        pass
    
    def define_metric(self, name, step_metric=None):
        pass