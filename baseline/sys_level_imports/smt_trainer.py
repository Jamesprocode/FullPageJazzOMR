import random

import lightning.pytorch as L
import torch

from sys_level_imports.dataset.eval_functions import compute_poliphony_metrics
from sys_level_imports.model.smt.configuration_smt import SMTConfig
from sys_level_imports.model.smt.modeling_smt import SMTModelForCausalLM
from torchinfo import summary
from sys_level_imports.dataset.tokenizer import untokenize
import gin
import wandb

from torch.nn import Conv1d

import transformers

@gin.configurable
class SMT_Trainer(L.LightningModule):
    def __init__(
        self,
        maxh,
        maxw,
        maxlen,
        out_categories,
        padding_token,
        in_channels,
        w2i,
        i2w,
        d_model=256,
        dim_ff=256,
        num_dec_layers=8,
        cp=None,
        texture="jazz",
        fold=None,
        lr=1e-4,
        warmup_steps=150,
        weight_decay=0.01,
        load_pretrained = False
    ):
        super().__init__()
        self.config = SMTConfig(
            maxh=maxh,
            maxw=maxw,
            maxlen=maxlen,
            out_categories=out_categories,
            padding_token=padding_token,
            in_channels=in_channels,
            w2i=w2i,
            i2w=i2w,
            d_model=d_model,
            dim_ff=dim_ff,
            attn_heads=4,
            num_dec_layers=num_dec_layers,
            use_flash_attn=True,
        )
        if load_pretrained:
            self.model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-grandstaff")
            # substitute the last layer with a new one of the correct size
            self.model.decoder.out_layer = Conv1d(d_model, out_categories, kernel_size=1)
            # update other config
            unpretrained_model = SMTModelForCausalLM(self.config)
            self.model.i2w = unpretrained_model.i2w.copy()
            self.model.w2i = unpretrained_model.w2i.copy()
            self.model.maxlen = unpretrained_model.maxlen
        else:
            self.model = SMTModelForCausalLM(self.config)
        self.padding_token = padding_token
        self.texture = texture
        self.fold = fold
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        self.preds = []
        self.grtrs = []

        self.save_hyperparameters()

        summary(
            self,
            input_size=[
                (1, 1, self.config.maxh, self.config.maxw),
                (1, self.config.maxlen),
            ],
            dtypes=[torch.float, torch.long],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW
        # only decay 2+-dimensional tensors, to exclude biases and norms
        # (filtering on dimensionality idea taken from Kaparthy's nano-GPT)
        params = [
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim >= 2
                ),
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim <= 1
                ),
                "weight_decay": 0,
            },
        ]

        optimizer = optimizer(params, lr=self.lr)

        self.lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
        )

        result = dict(optimizer=optimizer)
        result["lr_scheduler"] = {"scheduler": self.lr_scheduler, "interval": "step"}
        return result

    def forward(self, input, last_preds):
        return self.model(input, last_preds)

    def compute_loss(self, batch):
        (
            x,
            di,
            y,
            path_to_images
        ) = batch
        outputs = self.model(x, di[:, :-1], labels=y)
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        (
            x,
            di,
            y,
            path_to_images
        ) = batch
        loss = self.compute_loss(batch)

        self.log("train/loss", loss, on_epoch=True, batch_size=x.shape[0], prog_bar=True)

        if batch_idx == 0:
            # Create a wandb Table to log images with their paths
            table = wandb.Table(columns=["image", "path"])

            for i in range(x.shape[0]):
                image_tensor = x[i].squeeze().cpu()
                image_np = image_tensor.numpy()

                # Add image and its corresponding path to the table
                table.add_data(
                    wandb.Image(image_np),
                    path_to_images[i]
                )

            # Log the table to wandb
            self.logger.experiment.log({
                f"Input batch - Epoch {self.current_epoch}": table
            })

            # log also an image of the batch of images vertically stacked
            stacked_image = torch.cat([xx.squeeze().cpu() for xx in x], dim=0)
            wandb.log({
                f"Input batch image": wandb.Image(stacked_image.numpy())
            })

        return loss

    def validation_step(self, batch, batch_idx):
        (
            x,
            di,
            y,
            path_to_images
        ) = batch
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, on_epoch=True, batch_size=x.shape[0], prog_bar=True)

        self.predict_output(batch)

    def predict_output(self, batch):
        (
            x,
            di,
            y,
            path_to_images
        ) = batch

        for x_single,y_single in zip(x,y):
            predicted_sequence, _ = self.model.predict(input=x_single)

            dec = untokenize(predicted_sequence)
            gt = untokenize([self.model.i2w[token.item()] for token in y_single[:-1]])

            self.preds.append(dec)
            self.grtrs.append(gt)

    def compute_log_metrics(self, preds, grtrs, step="val"):
        cer, ser, ler = compute_poliphony_metrics(preds, grtrs)
        
        self.log(
            f"{step}/cer",
            100 if cer > 100 else cer,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{step}/ser",
            100 if ser > 100 else ser,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{step}/ler",
            100 if ler > 100 else ler,
            on_epoch=True,
            prog_bar=False,
        )


    def on_validation_epoch_end(self) -> None:
        self.compute_log_metrics(self.preds, self.grtrs, step="val")

        # random_index = random.randint(0, len(self.preds) - 1)
        random_index = 0 # always log the first, so we can see ho thing evolve over time
        predtoshow = self.preds[random_index]
        gttoshow = self.grtrs[random_index]
        # print(f"\n[Prediction] - {predtoshow}")
        # print(f"\n[GT] - {gttoshow}")

        # Create a wandb Table for predictions
        table = wandb.Table(columns=["Prediction", "Ground Truth"])
        table.add_data(predtoshow, gttoshow)
        
        # Log both table and text to wandb
        self.logger.experiment.log({
            f"val/predictions_table": table,
            f"val/example_prediction": wandb.Html(f"""
                <div style='white-space: pre-wrap;'>
                <b>Prediction:</b>\n{predtoshow}\n\n
                <b>Ground Truth:</b>\n{gttoshow}
                </div>
            """)
        })

        self.preds = []
        self.grtrs = []
        

    def test_step(self, batch, batch_idx):
        (
            x,
            di,
            y,
            path_to_images
        ) = batch
        loss = self.compute_loss(batch)
        self.log("test/loss", loss, on_epoch=True, batch_size=x.shape[0])

        self.predict_output(batch)

    def on_test_epoch_end(self) -> None:
        self.compute_log_metrics(self.preds, self.grtrs, step="test")
    
        import os

        os.makedirs(f"test_predictions/{self.texture}/{self.fold}/", exist_ok=True)
        # save all the test predictions in a folder with pairs of prediction and ground truth .kern files
        for i in range(len(self.preds)):
            with open(
                f"test_predictions/{self.texture}/{self.fold}/{i}_pred.kern", "w"
            ) as f:
                f.write(self.preds[i])
            with open(
                f"test_predictions/{self.texture}/{self.fold}/{i}_gt.kern", "w"
            ) as f:
                f.write(self.grtrs[i])

        self.preds = []
        self.grtrs = []
