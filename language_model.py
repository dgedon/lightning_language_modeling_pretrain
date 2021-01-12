from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    AutoConfig,
)
from data import LMDataModule


class LMModel(pl.LightningModule):
    def __init__(self,
                 model_name_or_path: str = 'bert-base-cased',
                 use_pretrained_model: bool = False,
                 learning_rate: float = 5e-5,
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.999,
                 adam_epsilon: float = 1e-8,
                 **kwargs
                 ):
        super().__init__()
        # save the hyperparameters. Can be accessed by self.hparams.[variable name]
        self.save_hyperparameters()

        if use_pretrained_model:
            config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
            self.model = AutoModelForMaskedLM.from_config(config=config)

    def forward(self, x):
        output = self.model(x)
        return output['logits']

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output['loss']
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.model(**batch)['loss']
        self.log('valid_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,
                          )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        return parser


def parse_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LMDataModule.add_model_specific_args(parser)
    parser = LMModel.add_model_specific_args(parser)
    return parser.parse_args()


def cli_main():
    # args
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="bert-base-cased")
    parser.add_argument('--use_pretrained_model', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234)
    args = parse_args(parser)

    # seed
    pl.seed_everything(args.seed)

    # data
    data_module = LMDataModule.from_argparse_args(args)

    # model
    model = LMModel(**vars(args))

    # training
    trainer = pl.Trainer.from_argparse_args(args) #, fast_dev_run=True)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    cli_main()
