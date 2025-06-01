from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score ,precision_score, recall_score

from neps.plot.tensorboard_eval import tblogger

class AutoML:

    def __init__(
        self,
        epochs: int,
        model: nn.Module,
        dataset_class,
        train_dataset: BaseVisionDataset,
        test_dataset: BaseVisionDataset,
        lr_scheduler: str,
        logger,
        dataset_reduction: float = 1.0,
        learning_rate: float = 10e-3,
        optimizer: str = "SarGD"
    ) -> None:
        
        # ========================================================
        ##################### Split the dataset ##################
        # ========================================================
        train_split, validation_split, classes_weights = train_val_splits(
            train_dataset=train_dataset,
            data_reduction_factor=dataset_reduction,
            train_percentage=0.80,
            logger=logger
        )

        self.logger = logger
        
        logger.info(f"Train reduction from {len(train_dataset) - len(validation_split)} to {len(train_split)}")

        batch_size = dataset_class.default_batch_size
        
        train_loader = DataLoader(
            dataset=train_split, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        validation_loader = DataLoader(
            dataset=validation_split, 
            batch_size=batch_size, 
            shuffle=True
        )
                
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if set(train_split.indices) & set(validation_split.indices):
            raise ValueError("Validation and Train split have common indices")
        
        self.lr_scheduler = lr_scheduler
        self.learning_rate = learning_rate
        self._model: nn.Module = model
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._optimizer = get_optimizer(optimizer, model, learning_rate, weight_decay=weight_decay) 
        
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        
        self.classes_weights = get_class_weights(train_split).to(device=self.device)

         
    def fit(self, enable_tensorboard=True) -> dict[str, Any]:
        # ==========================================================
        # ======================== Training ========================
        # ==========================================================
        start_time = time.time()
        
        criterion = nn.CrossEntropyLoss(weight=self.classes_weights)

        self._model.to(self.device)
        
        scheduler = get_lr_scheduler(self._optimizer, self.lr_scheduler, gamma=0.90)
        
        self._model.train()
        
        loss_per_epoch = []
        progress = tqdm(range(self.epochs * len(self.train_loader)))

        progress.set_description_str(f"Epoch {0}/{self.epochs}")            
        
        tqdm.write("Losses: []")

        for epoch in range(self.epochs):
            loss_per_batch = []
            for (data, target) in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self._optimizer.zero_grad()
                output = self._model(data)
                loss = criterion(output, target)
                loss.backward()
                loss_per_batch.append(loss.item())
                self._optimizer.step()
                progress.update(1)
            
            scheduler.step()
            loss_per_epoch.append(np.mean(loss_per_batch))
            
            progress.set_description_str(f"Epoch {epoch + 1}/{self.epochs}", refresh=True)            
            tqdm.write(f"Losses: {[i.round(5) for i in loss_per_epoch]}")
            
            ########## Start Tensorboard Logging
            if enable_tensorboard:
                tblogger.log(
                    loss=np.mean(loss_per_batch),
                    current_epoch=epoch,
                    write_summary_incumbent=False,  # live incumbent trajectory
                    writer_config_scalar=True,  # live loss trajectory for each config
                    writer_config_hparam=False,  # live parallel coordinate, scatter plot matrix, and table view
                    extra_data={
                        "lr_decay": tblogger.scalar_logging(value=scheduler.get_last_lr()[0]),
                        "train_loss": tblogger.scalar_logging(value=np.mean(loss_per_batch)),
                        "fidelity": tblogger.scalar_logging(value=self.epochs)
                    }
                )
            ########## End Tensorboard Logging
        progress.close()
        self._model.eval()
        train_loss = np.mean(loss_per_batch)
        validation_loss, validation_accuracy,f1, percison,recall = self.validate()
        self.logger.info(f"Validation Accuracy: {validation_accuracy}, F1: {f1}, Precision: {percison}, Recall: {recall}")
        cost = time.time() - start_time

        return {
            "loss": validation_loss,
            "validation_f1": f1,
            "validation_precision": percison,
            "validation_recall": recall,
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "train_loss": train_loss,
            "train_losses": loss_per_epoch,
            "cost": cost, 
            "fidelity": self.epochs,
        }

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        # ==========================================================
        # ======================== Testing =========================
        # ==========================================================    
        predictions = []
        labels = []
        self._model.to(self.device)
        self._model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                predicted = predicted.cpu()
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
    
    def validate(self) -> Tuple[np.ndarray, np.ndarray]:
        # ==========================================================
        # ======================= Validation =======================
        # ==========================================================            

        self._model.eval()
        predictions = []
        labels = []
        losses = []

        criterion = nn.CrossEntropyLoss() # weight=self.classes_weights

        for data, target in self.validation_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self._model(data)
            loss = criterion(output, target)
            loss.backward()
            losses.append(loss.item())
            
            predicted = torch.argmax(output, 1).cpu()
            labels.append(target.cpu().numpy())
            predictions.append(predicted.numpy())
    
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        self.logger.info(f"Validation Loss {np.mean(losses)}")
        
        scalarized_loss = np.mean(losses)
        
        validation_accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=np.nan)
        precision = precision_score(labels, predictions, average='weighted', zero_division=np.nan)
        recall = recall_score(labels, predictions, average='weighted', zero_division=np.nan)

        return scalarized_loss, validation_accuracy, f1, precision, recall


from sklearn.svm import SVR
from sklearn.base import BaseEstimator
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score

class SvrAutoML(BaseEstimator):
    def __init__(self, kernel_option, C_option, max_iter_option, epsilon_option):
        self.model = SVR(
            kernel=kernel_option, 
            C=C_option, 
            max_iter=max_iter_option, 
            epsilon=epsilon_option
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        y_preds = self.model.predict(X)
        mse = mean_squared_error(y_test, y_preds)
        rmse = root_mean_squared_error(y_test, y_preds)
        r2  = r2_score(y_test, y_preds)

        writer = tblogger.ConfigWriter(write_summary_incumbent=True)
        
        writer.add_scalar(tag="loss", scalar_value=rmse)
        writer.add_scalar(tag="mse", scalar_value=mse)
        writer.add_scalar(tag="r2", scalar_value=r2)
        writer.add_scalar(tag="C", scalar_value=self.model.C)
        writer.add_scalar(tag="max_iter", scalar_value=self.model.max_iter)
        writer.add_scalar(tag="epsilon", scalar_value=self.model.epsilon)

        writer.close()
        
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
